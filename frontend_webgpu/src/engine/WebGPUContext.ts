import { mat4, vec3 } from 'gl-matrix';
import type { IRenderContext } from './Renderer';
import type { TelemetryData } from '../lib/telemetry';
import { createSphere } from './geometry';

const wgslSource = `
struct Uniforms {
  modelViewMatrix : mat4x4<f32>,
  projectionMatrix : mat4x4<f32>,
  color: vec4<f32>,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> VertexOutput {
  var output : VertexOutput;
  output.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * vec4<f32>(position, 1.0);
  
  let depth = (output.position.z / output.position.w + 1.0) * 0.5;
  output.color = uniforms.color * vec4<f32>(vec3<f32>(1.2 - depth), 1.0);
  
  return output;
}

@fragment
fn fs_main(@location(0) color : vec4<f32>) -> @location(0) vec4<f32> {
  return color;
}
`;

export class WebGPUContext implements IRenderContext {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private pipeline!: GPURenderPipeline;

  private earthVBO!: GPUBuffer;
  private earthEBO!: GPUBuffer;
  private numEarthIndices = 0;

  private satVBO!: GPUBuffer;
  private satEBO!: GPUBuffer;
  private numSatIndices = 0;

  private depthTexture!: GPUTexture;

  private projMatrix = mat4.create();
  private viewMatrix = mat4.create();

  // Uniforms: [mat4(64), mat4(64), vec4(16)] = 144 bytes per object
  private earthUniformBuffer!: GPUBuffer;
  private earthBindGroup!: GPUBindGroup;

  private satUniformBuffer!: GPUBuffer;
  private satBindGroup!: GPUBindGroup;

  private satPosition = vec3.fromValues(0, 0, 0);
  private satColor = vec3.fromValues(0.2, 0.5, 1.0);

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

    // Create shader module
    const shaderModule = this.device.createShaderModule({ code: wgslSource });

    // Setup Depth Texture
    this.resizeDepthTexture(canvas.width, canvas.height);

    // Create Pipeline
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 3 * 4,
          attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }]
        }]
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }]
      },
      primitive: {
        topology: 'line-list', // Used for wireframe earth
        cullMode: 'back'
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus'
      }
    });

    // Earth Geometry
    const earthGeo = createSphere(6371, 32, 32);
    this.earthVBO = this.createBuffer(earthGeo.positions, GPUBufferUsage.VERTEX);
    this.earthEBO = this.createBuffer(earthGeo.indices, GPUBufferUsage.INDEX);
    this.numEarthIndices = earthGeo.indices.length;

    // Sat Geometry
    const satGeo = createSphere(150, 8, 8); // slightly larger for visibility
    this.satVBO = this.createBuffer(satGeo.positions, GPUBufferUsage.VERTEX);
    this.satEBO = this.createBuffer(satGeo.indices, GPUBufferUsage.INDEX);
    this.numSatIndices = satGeo.indices.length;

    // Uniform Buffers
    this.earthUniformBuffer = this.device.createBuffer({
      size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.satUniformBuffer = this.device.createBuffer({
      size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Bind Groups
    this.earthBindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.earthUniformBuffer } }]
    });

    this.satBindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.satUniformBuffer } }]
    });
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

    if (data.fdirMode === 2) { // Safe
      vec3.set(this.satColor, 1.0, 0.2, 0.2);
    } else if (data.fdirMode === 1) { // Degraded
      vec3.set(this.satColor, 1.0, 0.7, 0.1);
    } else if (data.payloadOn) { // Payload ON
      vec3.set(this.satColor, 0.1, 1.0, 0.4);
    } else { // Nominal
      vec3.set(this.satColor, 0.2, 0.5, 1.0);
    }
  }

  render(time: number): void {
    if (!this.device) return;

    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width !== this.depthTexture.width || canvas.height !== this.depthTexture.height) {
      this.resizeDepthTexture(canvas.width, canvas.height);
    }

    const aspect = canvas.width / canvas.height;
    mat4.perspective(this.projMatrix, 45 * Math.PI / 180, aspect, 100.0, 30000.0);

    const camAngle = time * 0.0001;
    const camRadius = 15000;
    mat4.lookAt(this.viewMatrix,
      [Math.cos(camAngle) * camRadius, 4000, Math.sin(camAngle) * camRadius],
      [0, 0, 0],
      [0, 1, 0]
    );

    // Update Earth Uniforms
    const earthModelView = mat4.clone(this.viewMatrix);
    mat4.rotateY(earthModelView, earthModelView, time * 0.00005);
    const earthUniformData = new Float32Array(36);
    earthUniformData.set(earthModelView, 0);
    earthUniformData.set(this.projMatrix, 16);
    earthUniformData.set([0.1, 0.2, 0.4, 1.0], 32); // color
    this.device.queue.writeBuffer(this.earthUniformBuffer, 0, earthUniformData);

    // Update Sat Uniforms
    const satModelView = mat4.clone(this.viewMatrix);
    mat4.translate(satModelView, satModelView, this.satPosition);
    // Expand sat rendering locally
    const satUniformData = new Float32Array(36);
    satUniformData.set(satModelView, 0);
    satUniformData.set(this.projMatrix, 16);
    satUniformData.set([...this.satColor, 1.0], 32);
    this.device.queue.writeBuffer(this.satUniformBuffer, 0, satUniformData);

    // Render Pass
    const commandEncoder = this.device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1.0 },
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

    renderPass.setPipeline(this.pipeline);

    // Draw Earth
    renderPass.setBindGroup(0, this.earthBindGroup);
    renderPass.setVertexBuffer(0, this.earthVBO);
    renderPass.setIndexBuffer(this.earthEBO, 'uint16');
    renderPass.drawIndexed(this.numEarthIndices);

    // Draw Satellite
    // Note: since our pipeline topology is 'line-list' (for wireframe earth),
    // the satellite will also render as line-list. In a real app we'd use multiple pipelines.
    // This gives it a cool wireframe look too.
    renderPass.setBindGroup(0, this.satBindGroup);
    renderPass.setVertexBuffer(0, this.satVBO);
    renderPass.setIndexBuffer(this.satEBO, 'uint16');
    renderPass.drawIndexed(this.numSatIndices);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(): void {
    this.device?.destroy();
  }
}
