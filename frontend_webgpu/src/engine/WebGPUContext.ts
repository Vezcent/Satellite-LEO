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
  earthRotation: f32,
  pad3: f32,
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
  @location(4) localNormal: vec3<f32>,
  @location(5) localPos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var output : VertexOutput;
  let pos = uniforms.modelViewMatrix * vec4<f32>(in.position, 1.0);
  output.position = uniforms.projectionMatrix * pos;
  output.viewPos = pos.xyz;
  output.normal = (uniforms.normalMatrix * vec4<f32>(in.normal, 0.0)).xyz;
  output.localNormal = in.normal;
  output.localPos = in.position;
  output.uv = in.uv;
  output.color = uniforms.color;
  return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  if (uniforms.isEarth > 1.4) {
    // MOON — procedural cratered surface
    let N = normalize(in.normal);
    let L = normalize(uniforms.sunDir.xyz);
    let diff = max(dot(N, L), 0.0);

    // Procedural craters using layered noise on local normal
    var p = in.localNormal * 8.0;
    var craters = 0.0;
    var amp = 1.0;
    for(var i: i32 = 0; i < 6; i++) {
        let n = sin(p.x * 1.7 + p.y * 2.3) * cos(p.y * 1.9 + p.z * 2.7) + sin(p.z * 2.1 + p.x * 1.3);
        let crater = smoothstep(0.6, 0.8, abs(n));
        craters += crater * amp;
        p *= 2.3;
        amp *= 0.45;
    }
    craters = clamp(craters / 2.5, 0.0, 1.0);

    // Maria (dark plains) vs Highlands
    let pm = in.localNormal * 2.0;
    let maria = sin(pm.x * 1.2) * cos(pm.y * 0.8 + pm.z * 1.5);
    let mariaFactor = smoothstep(-0.3, 0.3, maria);

    let highland = vec3<f32>(0.18, 0.17, 0.16);
    let mariaCol = vec3<f32>(0.08, 0.08, 0.07);
    var moonBase = mix(mariaCol, highland, mariaFactor);

    // Apply craters as darkening
    moonBase *= (0.7 + 0.3 * craters);

    // Direct lighting only — no atmosphere
    let moonColor = moonBase * (0.03 + 0.97 * diff);
    return vec4<f32>(moonColor, 1.0);

  } else if (uniforms.isEarth > 0.8) {
    let N = normalize(in.normal);
    let L = normalize(uniforms.sunDir.xyz);
    let V = normalize(-in.viewPos);
    let H = normalize(L + V);
    
    let diff = max(dot(N, L), 0.0);
    let spec = pow(max(dot(N, H), 0.0), 32.0) * 0.5;
    
    // PROCEDURAL CONTINENTS (Using Local Normal)
    var land = 0.0;
    var p_land = in.localNormal * 3.5;
    for(var i: i32 = 0; i < 5; i++) {
        land += (sin(p_land.x) * cos(p_land.y) + sin(p_land.y) * cos(p_land.z) + sin(p_land.z) * cos(p_land.x)) * pow(0.5, f32(i));
        p_land *= 2.2;
    }
    let isLand = land > 0.15;
    var earthColor = vec3<f32>(0.05, 0.15, 0.4); // Ocean
    if (isLand) { 
        earthColor = vec3<f32>(0.2, 0.4, 0.1) * (0.8 + 0.2 * sin(land * 10.0)); // Land
    }
    
    let litColor = earthColor * (0.1 + 0.9 * diff) + vec3<f32>(0.6, 0.8, 1.0) * spec * diff;
    
    // Dynamic SAA heatmap — compute geographic UV from local position
    let cosR = cos(-uniforms.earthRotation);
    let sinR = sin(-uniforms.earthRotation);
    let geoPos = vec3<f32>(
        in.localPos.x * cosR - in.localPos.z * sinR,
        in.localPos.y,
        in.localPos.x * sinR + in.localPos.z * cosR
    );
    let geoLat = asin(clamp(geoPos.y / length(geoPos), -1.0, 1.0));
    let geoLon = atan2(geoPos.z, geoPos.x);
    let geoUV = vec2<f32>(
        (geoLon + 3.14159) / (2.0 * 3.14159),
        1.0 - (geoLat + 1.5708) / 3.14159
    );
    let heat = textureSample(heatmapTexture, heatmapSampler, geoUV).r;
    
    var heatColor = vec3<f32>(0.0);
    if (heat > 0.01) {
        heatColor = colormap(heat) * 2.5;
        let contour = smoothstep(0.05, 0.0, abs(fract(heat * 15.0 + 0.5) - 0.5));
        heatColor += vec3<f32>(contour * 0.5);
    }
    
    let rim = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    let sunScattering = pow(max(dot(V, L), 0.0), 8.0);
    var atmColor = vec3<f32>(0.3, 0.6, 1.0) * rim * (0.2 + 0.8 * diff);
    atmColor += vec3<f32>(1.0, 0.9, 0.7) * sunScattering * rim * 2.0;
    
    let finalColor = mix(litColor, heatColor, heat * 0.8) + atmColor * (0.5 + uniforms.atmDensity * 2.0);
    return vec4<f32>(finalColor, 1.0);

  } else if (uniforms.isEarth < -0.5) {
    // STATIC SPARSE STARS
    let dir = normalize(in.viewPos);
    let grid = floor(dir * 1000.0);
    
    let s = fract(sin(dot(grid, vec3<f32>(12.9898, 78.233, 45.164))) * 43758.5453);
    let stars = pow(s, 800.0) * 20.0;
    
    return vec4<f32>(vec3<f32>(stars), 1.0);

  } else if (uniforms.isEarth > 0.4) {
    // Subtle Wireframe
    return vec4<f32>(in.color.rgb, 0.05); 
  } else if (uniforms.isEarth > 0.1) {
    // SUN — simple glowing disc, no rays
    let uv = in.uv * 2.0 - 1.0;
    let dist = length(uv);
    let core = smoothstep(0.2, 0.05, dist);
    let glow = exp(-dist * 3.0) * 1.2;
    let sunCol = vec3<f32>(1.0, 0.95, 0.85);
    return vec4<f32>(sunCol * (core * 2.0 + glow), core + glow);
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
  private sunPipeline!: GPURenderPipeline;

  private earthVBO!: GPUBuffer;
  private earthVBN!: GPUBuffer;
  private earthVBU!: GPUBuffer;
  private earthEBO!: GPUBuffer;
  private numEarthIndices = 0;

  private moonVBO!: GPUBuffer;
  private moonVBN!: GPUBuffer;
  private moonVBU!: GPUBuffer;
  private moonEBO!: GPUBuffer;
  private numMoonIndices = 0;

  private galaxyVBO!: GPUBuffer;
  private galaxyEBO!: GPUBuffer;
  private numGalaxyIndices = 0;

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
  private satUniformBuffers: GPUBuffer[] = [];
  private satBindGroups: GPUBindGroup[] = [];
  private sunUniformBuffer!: GPUBuffer;
  
  private earthBindGroup!: GPUBindGroup;
  private earthWireBindGroup!: GPUBindGroup;
  private moonBindGroup!: GPUBindGroup;
  private moonUniformBuffer!: GPUBuffer;
  private sunBindGroup!: GPUBindGroup;
  private galaxyBindGroup!: GPUBindGroup;
  private galaxyUniformBuffer!: GPUBuffer;

  private satPositions: vec3[] = [
    vec3.fromValues(0, 0, 0),
    vec3.fromValues(0, 0, 0),
    vec3.fromValues(0, 0, 0),
    vec3.fromValues(0, 0, 0)
  ];
  private satColors: vec3[] = [
    vec3.fromValues(0.23, 0.51, 0.96), // Sat 0: Blue
    vec3.fromValues(0.13, 0.77, 0.37), // Sat 1: Green
    vec3.fromValues(0.96, 0.62, 0.04), // Sat 2: Amber
    vec3.fromValues(0.66, 0.33, 0.97)  // Sat 3: Purple
  ];
  private activeSats: boolean[] = [false, false, false, false];
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

    this.sunPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: vertexState,
      fragment: {
        ...fragmentState,
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor },
            alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor }
          }
        }]
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { depthWriteEnabled: false, depthCompare: 'less', format: 'depth24plus' }
    });

    const earthGeo = createSphere(6371, 64, 64);
    this.earthVBO = this.createBuffer(earthGeo.positions, GPUBufferUsage.VERTEX);
    this.earthVBN = this.createBuffer(earthGeo.normals, GPUBufferUsage.VERTEX);
    this.earthVBU = this.createBuffer(earthGeo.uvs, GPUBufferUsage.VERTEX);
    this.earthEBO = this.createBuffer(earthGeo.indices, GPUBufferUsage.INDEX);
    this.numEarthIndices = earthGeo.indices.length;

    const galaxyGeo = createSphere(45000, 16, 16);
    this.galaxyVBO = this.createBuffer(galaxyGeo.positions, GPUBufferUsage.VERTEX);
    this.galaxyEBO = this.createBuffer(galaxyGeo.indices, GPUBufferUsage.INDEX);
    this.numGalaxyIndices = galaxyGeo.indices.length;

    const satGeo = createSphere(150, 8, 8);
    this.satVBO = this.createBuffer(satGeo.positions, GPUBufferUsage.VERTEX);
    this.satVBN = this.createBuffer(satGeo.normals, GPUBufferUsage.VERTEX);
    this.satVBU = this.createBuffer(satGeo.uvs, GPUBufferUsage.VERTEX);
    this.satEBO = this.createBuffer(satGeo.indices, GPUBufferUsage.INDEX);
    this.numSatIndices = satGeo.indices.length;

    const moonGeo = createSphere(1737, 32, 32); // Moon radius 1737 km
    this.moonVBO = this.createBuffer(moonGeo.positions, GPUBufferUsage.VERTEX);
    this.moonVBN = this.createBuffer(moonGeo.normals, GPUBufferUsage.VERTEX);
    this.moonVBU = this.createBuffer(moonGeo.uvs, GPUBufferUsage.VERTEX);
    this.moonEBO = this.createBuffer(moonGeo.indices, GPUBufferUsage.INDEX);
    this.numMoonIndices = moonGeo.indices.length;

    this.earthUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.earthWireUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.moonUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sunUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.heatmapSampler = this.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    await this.loadHeatmap();

    this.earthBindGroup = this.createBindGroup(this.earthUniformBuffer, this.solidPipeline);
    this.earthWireBindGroup = this.createBindGroup(this.earthWireUniformBuffer, this.wirePipeline);
    this.moonBindGroup = this.createBindGroup(this.moonUniformBuffer, this.solidPipeline);
    this.sunBindGroup = this.createBindGroup(this.sunUniformBuffer, this.sunPipeline);

    for (let i = 0; i < 4; i++) {
      const buf = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bg = this.createBindGroup(buf, this.solidPipeline);
      this.satUniformBuffers.push(buf);
      this.satBindGroups.push(bg);
    }

    this.galaxyUniformBuffer = this.device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.galaxyBindGroup = this.createBindGroup(this.galaxyUniformBuffer, this.solidPipeline);
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
    const satId = data.satId;
    if (satId < 0 || satId >= 4) return;

    this.activeSats[satId] = true;

    const latRad = data.latitudeDeg * Math.PI / 180;
    const lonRad = data.longitudeDeg * Math.PI / 180;
    const r = 6371 + data.altitudeKm;

    const x = r * Math.cos(latRad) * Math.cos(lonRad);
    const y = r * Math.cos(latRad) * Math.sin(lonRad);
    const z = r * Math.sin(latRad);

    this.satPositions[satId][0] = x;
    this.satPositions[satId][1] = z;
    this.satPositions[satId][2] = -y;

    if (data.isDone) {
      vec3.set(this.satColors[satId], 0.4, 0.4, 0.4);
    } else if (data.fdirMode === 2) {
      vec3.set(this.satColors[satId], 1.0, 0.2, 0.2);
    } else if (data.fdirMode === 1) {
      vec3.set(this.satColors[satId], 1.0, 0.7, 0.1);
    } else if (data.payloadOn) {
      vec3.set(this.satColors[satId], 0.1, 1.0, 0.4);
    } else {
      const defaultColors = [
        vec3.fromValues(0.23, 0.51, 0.96), // Sat 0: Blue
        vec3.fromValues(0.13, 0.77, 0.37), // Sat 1: Green
        vec3.fromValues(0.96, 0.62, 0.04), // Sat 2: Amber
        vec3.fromValues(0.66, 0.33, 0.97)  // Sat 3: Purple
      ];
      vec3.copy(this.satColors[satId], defaultColors[satId]);
    }

    this.currentAtmDensity = data.atmDensity;
  }

  render(time: number): void {
    if (!this.device) return;

    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width !== this.depthTexture.width || canvas.height !== this.depthTexture.height) {
      this.resizeDepthTexture(canvas.width, canvas.height);
    }

    const aspect = canvas.width / canvas.height;
    mat4.perspective(this.projMatrix, 45 * Math.PI / 180, aspect, 100.0, 120000.0);

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

    const updateUniform = (buffer: GPUBuffer, mv: mat4, color: vec3, isEarth: number, earthRot: number = 0) => {
      const data = new Float32Array(60); 
      data.set(mv, 0);
      data.set(this.projMatrix, 16);
      data.set(normalMatrix, 32);
      data.set([...color, 1.0], 48);
      data.set(sunDir, 52);
      data.set([isEarth, normDensity, earthRot, 0], 56);
      this.device.queue.writeBuffer(buffer, 0, data);
    };

    const earthRotation = time * 0.00005;
    updateUniform(this.earthUniformBuffer, earthModelView, vec3.fromValues(1,1,1), 1.0, earthRotation);
    const wireModelView = mat4.clone(earthModelView);
    mat4.scale(wireModelView, wireModelView, [1.002, 1.002, 1.002]);
    updateUniform(this.earthWireUniformBuffer, wireModelView, vec3.fromValues(0.15, 0.35, 0.6), 0.5);

    for (let i = 0; i < 4; i++) {
      if (!this.activeSats[i]) continue;
      const satModelView = mat4.clone(this.viewMatrix);
      mat4.translate(satModelView, satModelView, this.satPositions[i]);
      updateUniform(this.satUniformBuffers[i], satModelView, this.satColors[i], 0.0);
    }

    const galaxyView = mat4.clone(this.viewMatrix);
    galaxyView[12] = 0; galaxyView[13] = 0; galaxyView[14] = 0;
    updateUniform(this.galaxyUniformBuffer, galaxyView, vec3.fromValues(1,1,1), -1.0);

    // Moon orbit computation
    const moonOrbitRadius = 12000; // Closer orbit for visibility
    const moonAngle = time * 0.00003; // Slow orbit
    const moonX = Math.cos(moonAngle) * moonOrbitRadius;
    const moonZ = Math.sin(moonAngle) * moonOrbitRadius;
    const moonY = Math.sin(moonAngle * 0.5) * 2000; // Slight inclination

    const moonModelView = mat4.clone(this.viewMatrix);
    mat4.translate(moonModelView, moonModelView, [moonX, moonY, moonZ]);
    mat4.rotateY(moonModelView, moonModelView, -moonAngle); // Tidally locked

    const moonNormalMatrix = mat4.create();
    mat4.invert(moonNormalMatrix, moonModelView);
    mat4.transpose(moonNormalMatrix, moonNormalMatrix);

    // Moon uses its own normal matrix for correct lighting
    const updateMoonUniform = (buffer: GPUBuffer, mv: mat4, normMat: mat4) => {
      const data = new Float32Array(60);
      data.set(mv, 0);
      data.set(this.projMatrix, 16);
      data.set(normMat, 32);
      data.set([0.15, 0.15, 0.14, 1.0], 48);
      data.set(sunDir, 52);
      data.set([1.5, 0, 0, 0], 56); // isEarth = 1.5 for Moon
      this.device.queue.writeBuffer(buffer, 0, data);
    };
    updateMoonUniform(this.moonUniformBuffer, moonModelView, moonNormalMatrix as mat4);

    const sunWorldPos = [sunWorld[0] * 80000, sunWorld[1] * 80000, sunWorld[2] * 80000];
    const sunModelView = mat4.clone(this.viewMatrix);
    mat4.translate(sunModelView, sunModelView, sunWorldPos as any);
    sunModelView[0] = 1500; sunModelView[1] = 0; sunModelView[2] = 0;
    sunModelView[4] = 0; sunModelView[5] = 1500; sunModelView[6] = 0;
    sunModelView[8] = 0; sunModelView[9] = 0; sunModelView[10] = 1500;
    updateUniform(this.sunUniformBuffer, sunModelView, vec3.fromValues(1,1,1), 0.2);

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

    // 0. Galaxy Background
    renderPass.setPipeline(this.solidPipeline);
    renderPass.setBindGroup(0, this.galaxyBindGroup);
    renderPass.setVertexBuffer(0, this.galaxyVBO);
    renderPass.setVertexBuffer(1, this.earthVBN); // Dummy
    renderPass.setVertexBuffer(2, this.earthVBU); // Dummy
    renderPass.setIndexBuffer(this.galaxyEBO, 'uint16');
    renderPass.drawIndexed(this.numGalaxyIndices);

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

    // 1b. Moon
    renderPass.setPipeline(this.solidPipeline);
    renderPass.setBindGroup(0, this.moonBindGroup);
    renderPass.setVertexBuffer(0, this.moonVBO);
    renderPass.setVertexBuffer(1, this.moonVBN);
    renderPass.setVertexBuffer(2, this.moonVBU);
    renderPass.setIndexBuffer(this.moonEBO, 'uint16');
    renderPass.drawIndexed(this.numMoonIndices);

    renderPass.setPipeline(this.solidPipeline);
    renderPass.setVertexBuffer(0, this.satVBO);
    renderPass.setVertexBuffer(1, this.satVBN);
    renderPass.setVertexBuffer(2, this.satVBU);
    renderPass.setIndexBuffer(this.satEBO, 'uint16');
    for (let i = 0; i < 4; i++) {
      if (!this.activeSats[i]) continue;
      renderPass.setBindGroup(0, this.satBindGroups[i]);
      renderPass.drawIndexed(this.numSatIndices);
    }

    renderPass.setPipeline(this.sunPipeline);
    renderPass.setBindGroup(0, this.sunBindGroup);
    renderPass.setVertexBuffer(0, this.satVBO); // Reuse sphere for glow
    renderPass.setVertexBuffer(1, this.satVBN);
    renderPass.setVertexBuffer(2, this.satVBU);
    renderPass.setIndexBuffer(this.satEBO, 'uint16');
    renderPass.drawIndexed(this.numSatIndices);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(): void {}
}
