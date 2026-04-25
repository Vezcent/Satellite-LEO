import { mat4, vec3, vec4 } from 'gl-matrix';
import type { IRenderContext } from './Renderer';
import type { TelemetryData } from '../lib/telemetry';
import { createSphere } from './geometry';

const wgslSource = `
struct Uniforms {
  modelViewMatrix : mat4x4<f32>,
  projectionMatrix : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
  color: vec4<f32>,
  sunDir: vec4<f32>,
  isEarth: f32,
  atmDensity: f32,
  pad2: f32, pad3: f32,
};

fn colormap(x: f32) -> vec3<f32> {
    let r = clamp(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0);
    let g = clamp(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0);
    let b = clamp(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var heatmapSampler : sampler;
@binding(2) @group(0) var heatmapTexture : texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) viewPos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var output : VertexOutput;
  let pos = uniforms.modelViewMatrix * vec4<f32>(in.position, 1.0);
  output.position = uniforms.projectionMatrix * pos;
  output.viewPos = pos.xyz;
  output.normal = (uniforms.normalMatrix * vec4<f32>(in.normal, 0.0)).xyz;
  output.uv = in.uv;
  output.color = uniforms.color;
  return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  if (uniforms.isEarth > 0.8) {
    let N = normalize(in.normal);
    let L = normalize(uniforms.sunDir.xyz);
    let diff = max(dot(N, L), 0.0);
    
    let earthBase = vec3<f32>(0.1, 0.2, 0.5);
    let litColor = earthBase * (0.4 + 0.6 * diff);
    
    let uvFlipped = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    let heat = textureSample(heatmapTexture, heatmapSampler, uvFlipped).r;
    
    var heatColor = vec3<f32>(0.0);
    if (heat > 0.01) {
        heatColor = colormap(heat) * 1.5;
        let contour = smoothstep(0.05, 0.0, abs(fract(heat * 10.0 + 0.5) - 0.5));
        heatColor += vec3<f32>(contour * 0.3);
    }
    
    let viewDir = normalize(-in.viewPos);
    let fresnel = pow(1.0 - max(dot(N, viewDir), 0.0), 3.0);
    let atmColor = vec3<f32>(0.4, 0.7, 1.0) * fresnel * (0.5 + uniforms.atmDensity * 3.0);
    
    var finalColor = litColor;
    if (heat > 0.1) {
        finalColor = mix(litColor, heatColor, heat * 0.8);
    } else {
        finalColor += heatColor * 0.5;
    }
    
    return vec4<f32>(finalColor + atmColor, 1.0);
  } else if (uniforms.isEarth > 0.2) {
    return vec4<f32>(in.color.rgb, 0.2);
  } else {
    return in.color;
  }
}
`;

export class WebGPUContext implements IRenderContext {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private solidPipeline!: GPURenderPipeline;
  private wirePipeline!: GPURenderPipeline;

  private earthVBO!: GPUBuffer;
  private earthVBN!: GPUBuffer;
  private earthVBU!: GPUBuffer;
  private earthEBO!: GPUBuffer;
  private numEarthIndices = 0;

  private satVBO!: GPUBuffer;
  private satVBN!: GPUBuffer;
  private satVBU!: GPUBuffer;
  private satEBO!: GPUBuffer;
  private numSatIndices = 0;

  private depthTexture!: GPUTexture;
  private heatmapTexture!: GPUTexture;
  private heatmapSampler!: GPUSampler;

  private projMatrix = mat4.create();
  private viewMatrix = mat4.create();

  private earthUniformBuffer!: GPUBuffer;
  private earthWireUniformBuffer!: GPUBuffer;
  private satUniformBuffer!: GPUBuffer;
  
  private earthBindGroup!: GPUBindGroup;
  private earthWireBindGroup!: GPUBindGroup;
  private satBindGroup!: GPUBindGroup;

  private satPosition = vec3.fromValues(0, 0, 0);
  private satColor = vec3.fromValues(0.2, 0.5, 1.0);
  private currentAtmDensity = 0;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPUAdapter found');
    this.device = await adapter.requestDevice();

    this.context = canvas.getContext('webgpu') as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied'
    });

    const shaderModule = this.device.createShaderModule({ code: wgslSource });
    this.resizeDepthTexture(canvas.width, canvas.height);

    const vertexState = {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        { arrayStride: 12, attributes: [{ format: 'float32x3' as GPUVertexFormat, offset: 0, shaderLocation: 0 }] },
        { arrayStride: 12, attributes: [{ format: 'float32x3' as GPUVertexFormat, offset: 0, shaderLocation: 1 }] },
        { arrayStride: 8, attributes: [{ format: 'float32x2' as GPUVertexFormat, offset: 0, shaderLocation: 2 }] }
      ]
    };

    const fragmentState = {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{
        format: this.format,
        blend: {
          color: { srcFactor: 'src-alpha' as GPUBlendFactor, dstFactor: 'one-minus-src-alpha' as GPUBlendFactor },
          alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one-minus-src-alpha' as GPUBlendFactor }
        }
      }]
    };

    this.solidPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: vertexState,
      fragment: fragmentState,
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
    });

    this.wirePipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: vertexState,
      fragment: fragmentState,
      primitive: { topology: 'line-list' },
      depthStencil: { depthWriteEnabled: false, depthCompare: 'less', format: 'depth24plus' }
    });

    const earthGeo = createSphere(6371, 32, 32);
    this.earthVBO = this.createBuffer(earthGeo.positions, GPUBufferUsage.VERTEX);
    this.earthVBN = this.createBuffer(earthGeo.normals, GPUBufferUsage.VERTEX);
    this.earthVBU = this.createBuffer(earthGeo.uvs, GPUBufferUsage.VERTEX);
    this.earthEBO = this.createBuffer(earthGeo.indices, GPUBufferUsage.INDEX);
    this.numEarthIndices = earthGeo.indices.length;

    const satGeo = createSphere(150, 8, 8);
    this.satVBO = this.createBuffer(satGeo.positions, GPUBufferUsage.VERTEX);
    this.satVBN = this.createBuffer(satGeo.normals, GPUBufferUsage.VERTEX);
    this.satVBU = this.createBuffer(satGeo.uvs, GPUBufferUsage.VERTEX);
    this.satEBO = this.createBuffer(satGeo.indices, GPUBufferUsage.INDEX);
    this.numSatIndices = satGeo.indices.length;

    this.earthUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.earthWireUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.satUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.heatmapSampler = this.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    await this.loadHeatmap();

    this.earthBindGroup = this.createBindGroup(this.earthUniformBuffer, this.solidPipeline);
    this.earthWireBindGroup = this.createBindGroup(this.earthWireUniformBuffer, this.wirePipeline);
    this.satBindGroup = this.createBindGroup(this.satUniformBuffer, this.solidPipeline);
  }

  private createBindGroup(buffer: GPUBuffer, pipeline: GPURenderPipeline): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: this.heatmapSampler },
        { binding: 2, resource: this.heatmapTexture.createView() }
      ]
    });
  }

  private async loadHeatmap() {
    this.heatmapTexture = this.device.createTexture({
      size: [121, 90, 1],
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    try {
      const res = await fetch('/data/saa_heatmap_600km.csv');
      const text = await res.text();
      const lines = text.trim().split(/\r?\n/).slice(1);
      const w = 121, h = 90;
      const data = new Float32Array(w * h);
      let maxFlux = 0;
      for (let i = 0; i < lines.length && i < w*h; i++) {
        const parts = lines[i].split(',');
        if (parts.length >= 4) {
          const flux = parseFloat(parts[2]) + parseFloat(parts[3]);
          data[i] = flux;
          if (flux > maxFlux) maxFlux = flux;
        }
      }
      const ui8Data = new Uint8Array(w * h);
      if (maxFlux > 0) {
        for (let i = 0; i < data.length; i++) {
          const val = data[i] / maxFlux;
          ui8Data[i] = Math.floor(Math.pow(val, 0.3) * 255);
        }
      }
      this.device.queue.writeTexture({ texture: this.heatmapTexture }, ui8Data, { bytesPerRow: w }, [w, h, 1]);
    } catch (e) {
      console.error("Failed to load heatmap", e);
    }
  }

  private resizeDepthTexture(width: number, height: number) {
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
  }

  private createBuffer(data: Float32Array | Uint16Array, usage: GPUBufferUsageFlags) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    if (data instanceof Float32Array) {
      new Float32Array(buffer.getMappedRange()).set(data);
    } else {
      new Uint16Array(buffer.getMappedRange()).set(data);
    }
    buffer.unmap();
    return buffer;
  }

  updateTelemetry(data: TelemetryData): void {
    const latRad = data.latitudeDeg * Math.PI / 180;
    const lonRad = data.longitudeDeg * Math.PI / 180;
    const r = 6371 + data.altitudeKm;

    const x = r * Math.cos(latRad) * Math.cos(lonRad);
    const y = r * Math.cos(latRad) * Math.sin(lonRad);
    const z = r * Math.sin(latRad);

    this.satPosition[0] = x;
    this.satPosition[1] = z;
    this.satPosition[2] = -y;

    if (data.fdirMode === 2) { vec3.set(this.satColor, 1.0, 0.2, 0.2); }
    else if (data.fdirMode === 1) { vec3.set(this.satColor, 1.0, 0.7, 0.1); }
    else if (data.payloadOn) { vec3.set(this.satColor, 0.1, 1.0, 0.4); }
    else { vec3.set(this.satColor, 0.2, 0.5, 1.0); }

    this.currentAtmDensity = data.atmDensity;
  }

  render(time: number): void {
    if (!this.device) return;

    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width !== this.depthTexture.width || canvas.height !== this.depthTexture.height) {
      this.resizeDepthTexture(canvas.width, canvas.height);
    }

    const aspect = canvas.width / canvas.height;
    mat4.perspective(this.projMatrix, 45 * Math.PI / 180, aspect, 100.0, 50000.0);

    const camAngle = time * 0.0001;
    const camRadius = 25000;
    mat4.lookAt(this.viewMatrix,
      [Math.cos(camAngle) * camRadius, 5000, Math.sin(camAngle) * camRadius],
      [0, 0, 0],
      [0, 1, 0]
    );

    const sunWorld = vec4.fromValues(1.0, 0.5, 0.8, 0.0);
    const sunView = vec4.create();
    vec4.transformMat4(sunView, sunWorld, this.viewMatrix);
    const sunDir = new Float32Array([sunView[0], sunView[1], sunView[2], 0.0]);

    const normDensity = Math.min(Math.max((Math.log10(this.currentAtmDensity + 1e-20) + 15) / 5.0, 0.0), 1.0);

    // Update Earth Uniforms
    const earthModelView = mat4.clone(this.viewMatrix);
    mat4.rotateY(earthModelView, earthModelView, time * 0.00005);
    const normalMatrix = mat4.create();
    mat4.invert(normalMatrix, earthModelView);
    mat4.transpose(normalMatrix, normalMatrix);

    const updateUniform = (buffer: GPUBuffer, mv: mat4, color: vec3, isEarth: number) => {
      const data = new Float32Array(60); 
      data.set(mv, 0);
      data.set(this.projMatrix, 16);
      data.set(normalMatrix, 32);
      data.set([...color, 1.0], 48);
      data.set(sunDir, 52);
      data.set([isEarth, normDensity, 0, 0], 56);
      this.device.queue.writeBuffer(buffer, 0, data);
    };

    updateUniform(this.earthUniformBuffer, earthModelView, vec3.fromValues(1,1,1), 1.0);
    const wireModelView = mat4.clone(earthModelView);
    mat4.scale(wireModelView, wireModelView, [1.002, 1.002, 1.002]);
    updateUniform(this.earthWireUniformBuffer, wireModelView, vec3.fromValues(0.15, 0.35, 0.6), 0.5);

    const satModelView = mat4.clone(this.viewMatrix);
    mat4.translate(satModelView, satModelView, this.satPosition);
    updateUniform(this.satUniformBuffer, satModelView, this.satColor, 0.0);

    const commandEncoder = this.device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.01, g: 0.01, b: 0.02, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      }
    });

    renderPass.setPipeline(this.solidPipeline);
    renderPass.setBindGroup(0, this.earthBindGroup);
    renderPass.setVertexBuffer(0, this.earthVBO);
    renderPass.setVertexBuffer(1, this.earthVBN);
    renderPass.setVertexBuffer(2, this.earthVBU);
    renderPass.setIndexBuffer(this.earthEBO, 'uint16');
    renderPass.drawIndexed(this.numEarthIndices);

    renderPass.setPipeline(this.wirePipeline);
    renderPass.setBindGroup(0, this.earthWireBindGroup);
    renderPass.setVertexBuffer(0, this.earthVBO);
    renderPass.setVertexBuffer(1, this.earthVBN);
    renderPass.setVertexBuffer(2, this.earthVBU);
    renderPass.setIndexBuffer(this.earthEBO, 'uint16');
    renderPass.drawIndexed(this.numEarthIndices);

    renderPass.setPipeline(this.solidPipeline);
    renderPass.setBindGroup(0, this.satBindGroup);
    renderPass.setVertexBuffer(0, this.satVBO);
    renderPass.setVertexBuffer(1, this.satVBN);
    renderPass.setVertexBuffer(2, this.satVBU);
    renderPass.setIndexBuffer(this.satEBO, 'uint16');
    renderPass.drawIndexed(this.numSatIndices);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(): void {}
}
