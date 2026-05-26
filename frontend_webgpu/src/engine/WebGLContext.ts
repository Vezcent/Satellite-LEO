import { mat4, vec3, vec4 } from 'gl-matrix';
import type { IRenderContext } from './Renderer';
import type { TelemetryData } from '../lib/telemetry';
import { createSphere } from './geometry';

const vsSource = `#version 300 es
layout(location = 0) in vec3 aVertexPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uNormalMatrix;
uniform vec3 uColor;

out vec3 vColor;
out vec3 vNormal;
out vec3 vLocalNormal;
out vec2 vUV;
out vec3 vPosition;
out vec3 vLocalPos;

void main() {
  vec4 pos = uModelViewMatrix * vec4(aVertexPosition, 1.0);
  gl_Position = uProjectionMatrix * pos;
  vPosition = pos.xyz;
  vNormal = mat3(uNormalMatrix) * aNormal;
  vLocalNormal = aNormal;
  vLocalPos = aVertexPosition;
  vUV = aUV;
  vColor = uColor;
}
`;

const fsSource = `#version 300 es
precision highp float;

in vec3 vColor;
in vec3 vNormal;
in vec3 vLocalNormal;
in vec2 vUV;
in vec3 vPosition;
in vec3 vLocalPos;

uniform vec3 uSunDir;
uniform sampler2D uHeatmap;
uniform float uIsEarth; // 1.0 = solid, 0.5 = wireframe, 0.0 = sat
uniform float uAtmDensity;
uniform float uEarthRotation; // current Earth rotation angle (radians)

vec3 colormap(float x) {
    float r = clamp(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0);
    return vec3(r, g, b);
}

out vec4 fragColor;

void main() {
  if (uIsEarth > 0.8) {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uSunDir);
    vec3 V = normalize(-vPosition);
    vec3 H = normalize(L + V);
    
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.5; // Ocean glint
    
    // PROCEDURAL CONTINENTS (Using Local Normal for rotation)
    float land = 0.0;
    vec3 p_land = vLocalNormal * 3.5;
    for(int i=0; i<5; i++) {
        land += (sin(p_land.x) * cos(p_land.y) + sin(p_land.y) * cos(p_land.z) + sin(p_land.z) * cos(p_land.x)) * pow(0.5, float(i));
        p_land *= 2.2;
    }
    bool isLand = land > 0.15;
    vec3 earthColor = isLand ? vec3(0.2, 0.4, 0.1) : vec3(0.05, 0.15, 0.4);
    if (isLand) { earthColor *= (0.8 + 0.2 * sin(land * 10.0)); } // Texture detail
    
    vec3 litColor = earthColor * (0.1 + 0.9 * diff) + vec3(0.6, 0.8, 1.0) * spec * diff;
    
    // Dynamic SAA heatmap — compute geographic UV from local position
    // Undo Earth rotation to get true geographic coordinates
    float cosR = cos(-uEarthRotation);
    float sinR = sin(-uEarthRotation);
    vec3 geoPos = vec3(
        vLocalPos.x * cosR - vLocalPos.z * sinR,
        vLocalPos.y,
        vLocalPos.x * sinR + vLocalPos.z * cosR
    );
    float geoLat = asin(clamp(geoPos.y / length(geoPos), -1.0, 1.0));
    float geoLon = atan(geoPos.z, geoPos.x);
    // Map lat [-π/2, π/2] → [0, 1], lon [-π, π] → [0, 1]
    vec2 geoUV = vec2(
        (geoLon + 3.14159) / (2.0 * 3.14159),
        1.0 - (geoLat + 1.5708) / 3.14159
    );
    float heat = texture(uHeatmap, geoUV).r;
    vec3 heatColor = vec3(0.0);
    if (heat > 0.01) {
        heatColor = colormap(heat) * 2.5;
        float contour = smoothstep(0.05, 0.0, abs(fract(heat * 15.0 + 0.5) - 0.5));
        heatColor += vec3(contour * 0.5);
    }
    
    // Advanced Atmospheric Scattering (Rayleigh approximation)
    float rim = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    float sunScattering = pow(max(dot(V, L), 0.0), 8.0);
    vec3 atmColor = vec3(0.3, 0.6, 1.0) * rim * (0.2 + 0.8 * diff);
    atmColor += vec3(1.0, 0.9, 0.7) * sunScattering * rim * 2.0; // Bloom near sun
    
    vec3 finalColor = mix(litColor, heatColor, heat * 0.8) + atmColor * (0.5 + uAtmDensity * 2.0);
    fragColor = vec4(finalColor, 1.0);

  } else if (uIsEarth < -0.5) {
    // STATIC SPARSE STARS
    vec3 dir = normalize(vPosition);
    vec3 grid = floor(dir * 1000.0); // Snap to high-res grid for stability
    
    float s = fract(sin(dot(grid, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
    float stars = pow(s, 800.0) * 20.0; // Extremely sparse
    
    fragColor = vec4(vec3(stars), 1.0);

  } else if (uIsEarth > 0.4) {
    // Subtle Wireframe
    fragColor = vec4(vColor, 0.05); 
  } else if (uIsEarth > 0.1) {
    // SUN STARBURST (Billboard pass)
    vec2 uv = vUV * 2.0 - 1.0;
    float dist = length(uv);
    float glow = exp(-dist * 4.0) * 1.5;
    float rays = 0.0;
    for(int i=0; i<8; i++) {
        float angle = float(i) * 3.14159 / 4.0;
        vec2 dir = vec2(cos(angle), sin(angle));
        float r = pow(max(dot(normalize(uv), dir), 0.0), 40.0);
        rays += r * exp(-dist * 1.5);
    }
    float core = smoothstep(0.15, 0.05, dist);
    vec3 sunCol = vec3(1.0, 0.95, 0.8);
    fragColor = vec4(sunCol * (glow + rays * 0.6 + core * 2.0), glow + rays + core);
  } else {
    fragColor = vec4(vColor, 1.0); // Satellite
  }
}
`;

export class WebGLContext implements IRenderContext {
  private gl!: WebGL2RenderingContext;
  private program!: WebGLProgram;
  
  private vaoEarth!: WebGLVertexArrayObject;
  private numEarthIndices = 0;
  private vaoGalaxy!: WebGLVertexArrayObject;
  private numGalaxyIndices = 0;
  private vaoSat!: WebGLVertexArrayObject;
  private numSatIndices = 0;

  private projectionMatrix = mat4.create();
  private viewMatrix = mat4.create();
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

  // Uniform locations
  private uProj!: WebGLUniformLocation;
  private uModelView!: WebGLUniformLocation;
  private uNormalMatrix!: WebGLUniformLocation;
  private uColor!: WebGLUniformLocation;
  private uSunDir!: WebGLUniformLocation;
  private uHeatmap!: WebGLUniformLocation;
  private uIsEarth!: WebGLUniformLocation;
  private uAtmDensity!: WebGLUniformLocation;
  private uEarthRotation!: WebGLUniformLocation;
  
  private heatmapTex!: WebGLTexture;
  private currentEarthRotation = 0;

  // ── Telemetry Trails (orbital paths) ──
  private trailBuffers: Float32Array[] = [
    new Float32Array(200 * 3),
    new Float32Array(200 * 3),
    new Float32Array(200 * 3),
    new Float32Array(200 * 3)
  ];
  private trailCounts = [0, 0, 0, 0];
  private trailHeads = [0, 0, 0, 0];
  private trailVBOs: WebGLBuffer[] = [];
  private trailVAOs: WebGLVertexArrayObject[] = [];

  // ── Target Orbit Ring ──
  private targetOrbitVAO!: WebGLVertexArrayObject;
  private targetOrbitVBO!: WebGLBuffer;
  private targetAltitudeKm = 600;
  private static readonly ORBIT_SEGMENTS = 64;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('WebGL2 not supported');
    this.gl = gl;

    const vertShader = this.compileShader(gl.VERTEX_SHADER, vsSource);
    const fragShader = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
    
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vertShader);
    gl.attachShader(this.program, fragShader);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Shader linking failed: ' + gl.getProgramInfoLog(this.program));
    }

    this.uProj = gl.getUniformLocation(this.program, 'uProjectionMatrix')!;
    this.uModelView = gl.getUniformLocation(this.program, 'uModelViewMatrix')!;
    this.uNormalMatrix = gl.getUniformLocation(this.program, 'uNormalMatrix')!;
    this.uColor = gl.getUniformLocation(this.program, 'uColor')!;
    this.uSunDir = gl.getUniformLocation(this.program, 'uSunDir')!;
    this.uHeatmap = gl.getUniformLocation(this.program, 'uHeatmap')!;
    this.uIsEarth = gl.getUniformLocation(this.program, 'uIsEarth')!;
    this.uAtmDensity = gl.getUniformLocation(this.program, 'uAtmDensity')!;
    this.uEarthRotation = gl.getUniformLocation(this.program, 'uEarthRotation')!

    const earthGeo = createSphere(6371, 64, 64); // Increased segments for smoother specular
    this.vaoEarth = this.createVao(earthGeo.positions, earthGeo.normals, earthGeo.uvs, earthGeo.indices);
    this.numEarthIndices = earthGeo.indices.length;

    const galaxyGeo = createSphere(45000, 16, 16);
    this.vaoGalaxy = this.createVao(galaxyGeo.positions, galaxyGeo.normals, galaxyGeo.uvs, galaxyGeo.indices);
    this.numGalaxyIndices = galaxyGeo.indices.length;

    const satGeo = createSphere(100, 8, 8);
    this.vaoSat = this.createVao(satGeo.positions, satGeo.normals, satGeo.uvs, satGeo.indices);
    this.numSatIndices = satGeo.indices.length;

    // Hack: use vaoSat logic for simplicity but it's a quad
    this.numSatIndices = satGeo.indices.length; 
    // I'll just draw the sun using the Earth VAO or Sat VAO for now, 
    // but better to have a dedicated one. I'll stick to a simple approach.

    // ── Trail VBOs/VAOs (dynamic line strip) ──
    for (let i = 0; i < 4; i++) {
      const vbo = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, this.trailBuffers[i], gl.DYNAMIC_DRAW);
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      gl.enableVertexAttribArray(0); // aVertexPosition
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      
      this.trailVBOs.push(vbo);
      this.trailVAOs.push(vao);
    }

    // ── Target Orbit Ring VBO/VAO ──
    this.buildTargetOrbit(gl);

    this.loadHeatmap();

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  /** Build a circle of ORBIT_SEGMENTS points at targetAltitudeKm in the XZ plane */
  private buildTargetOrbit(gl: WebGL2RenderingContext) {
    const r = 6371 + this.targetAltitudeKm;
    const N = WebGLContext.ORBIT_SEGMENTS;
    const verts = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const angle = (i / N) * 2 * Math.PI;
      verts[i * 3]     = r * Math.cos(angle); // x
      verts[i * 3 + 1] = 0;                    // y (equatorial plane)
      verts[i * 3 + 2] = r * Math.sin(angle); // z
    }
    if (!this.targetOrbitVBO) {
      this.targetOrbitVBO = gl.createBuffer()!;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.targetOrbitVBO);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    if (!this.targetOrbitVAO) {
      this.targetOrbitVAO = gl.createVertexArray()!;
    }
    gl.bindVertexArray(this.targetOrbitVAO);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  /** Update target altitude and rebuild orbit ring geometry */
  setTargetAltitude(altKm: number) {
    this.targetAltitudeKm = altKm;
    if (this.gl) this.buildTargetOrbit(this.gl);
  }

  private loadHeatmap(): void {
    const gl = this.gl;
    this.heatmapTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.heatmapTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    fetch('/data/saa_heatmap_600km.csv')
      .then(r => r.text())
      .then(text => {
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
            ui8Data[i] = Math.floor(Math.pow(val, 0.3) * 255); // even more aggressive gamma
          }
        }
        gl.bindTexture(gl.TEXTURE_2D, this.heatmapTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, ui8Data);
      })
      .catch(e => console.error("Failed to load heatmap", e));
  }

  private compileShader(type: number, source: string): WebGLShader {
    const shader = this.gl.createShader(type)!;
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      throw new Error('Shader compile failed: ' + this.gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  private createVao(positions: Float32Array, normals: Float32Array, uvs: Float32Array, indices: Uint16Array): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const vboN = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vboN);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const vboU = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vboU);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    return vao;
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

    // Update trail ring buffer
    const idx = this.trailHeads[satId] * 3;
    this.trailBuffers[satId][idx] = this.satPositions[satId][0];
    this.trailBuffers[satId][idx + 1] = this.satPositions[satId][1];
    this.trailBuffers[satId][idx + 2] = this.satPositions[satId][2];
    this.trailHeads[satId] = (this.trailHeads[satId] + 1) % 200;
    if (this.trailCounts[satId] < 200) this.trailCounts[satId]++;

    // Upload trail data
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trailVBOs[satId]);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.trailBuffers[satId]);
  }

  render(time: number): void {
    const gl = this.gl;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.01, 0.01, 0.02, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.program);

    // 0. Galaxy Background (Draw first, no depth write)
    gl.disable(gl.DEPTH_TEST);
    const aspect = gl.canvas.width / gl.canvas.height;
    mat4.perspective(this.projectionMatrix, 45 * Math.PI / 180, aspect, 100.0, 100000.0);
    gl.uniformMatrix4fv(this.uProj, false, this.projectionMatrix);

    const galaxyView = mat4.clone(this.viewMatrix);
    galaxyView[12] = 0; galaxyView[13] = 0; galaxyView[14] = 0; // Center galaxy on camera
    gl.uniformMatrix4fv(this.uModelView, false, galaxyView);
    gl.uniform1f(this.uIsEarth, -1.0);
    gl.bindVertexArray(this.vaoGalaxy);
    gl.drawElements(gl.TRIANGLES, this.numGalaxyIndices, gl.UNSIGNED_SHORT, 0);
    gl.enable(gl.DEPTH_TEST);

    const camAngle = time * 0.0001;
    const camRadius = 25000;
    mat4.lookAt(this.viewMatrix, 
      [Math.cos(camAngle) * camRadius, 5000, Math.sin(camAngle) * camRadius],
      [0, 0, 0],
      [0, 1, 0]
    );

    // Transform sun direction into view space
    const sunWorld = vec4.fromValues(1.0, 0.5, 0.8, 0.0);
    const sunView = vec4.create();
    vec4.transformMat4(sunView, sunWorld, this.viewMatrix);
    gl.uniform3f(this.uSunDir, sunView[0], sunView[1], sunView[2]);

    const normDensity = Math.min(Math.max((Math.log10(this.currentAtmDensity + 1e-20) + 15) / 5.0, 0.0), 1.0);
    gl.uniform1f(this.uAtmDensity, normDensity);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.heatmapTex);
    gl.uniform1i(this.uHeatmap, 0);

    // 1. Earth
    const earthModelView = mat4.clone(this.viewMatrix);
    this.currentEarthRotation = time * 0.00005;
    mat4.rotateY(earthModelView, earthModelView, this.currentEarthRotation);
    gl.uniformMatrix4fv(this.uModelView, false, earthModelView);
    
    const normalMatrix = mat4.create();
    mat4.invert(normalMatrix, earthModelView);
    mat4.transpose(normalMatrix, normalMatrix);
    gl.uniformMatrix4fv(this.uNormalMatrix, false, normalMatrix);

    // Pass Earth rotation to shader for dynamic heatmap geo-referencing
    gl.uniform1f(this.uEarthRotation, this.currentEarthRotation);

    gl.bindVertexArray(this.vaoEarth);

    gl.uniform1f(this.uIsEarth, 1.0);
    gl.enable(gl.CULL_FACE);
    gl.drawElements(gl.TRIANGLES, this.numEarthIndices, gl.UNSIGNED_SHORT, 0);

    const wireModelView = mat4.clone(earthModelView);
    mat4.scale(wireModelView, wireModelView, [1.002, 1.002, 1.002]);
    gl.uniformMatrix4fv(this.uModelView, false, wireModelView);
    gl.uniform1f(this.uIsEarth, 0.5);
    gl.uniform3f(this.uColor, 0.15, 0.35, 0.6);
    gl.disable(gl.CULL_FACE);
    gl.drawElements(gl.LINES, this.numEarthIndices, gl.UNSIGNED_SHORT, 0);

    // 2. Satellites
    gl.bindVertexArray(this.vaoSat);
    gl.enable(gl.CULL_FACE);
    gl.uniform1f(this.uIsEarth, 0.0);

    for (let i = 0; i < 4; i++) {
      if (!this.activeSats[i]) continue;
      const satModelView = mat4.clone(this.viewMatrix);
      mat4.translate(satModelView, satModelView, this.satPositions[i]);
      gl.uniformMatrix4fv(this.uModelView, false, satModelView);
      gl.uniform3fv(this.uColor, this.satColors[i]);
      gl.drawElements(gl.TRIANGLES, this.numSatIndices, gl.UNSIGNED_SHORT, 0);
    }

    // 3. Sun Billboard (The "Light Rays")
    const sunWorldPos = [sunWorld[0] * 40000, sunWorld[1] * 40000, sunWorld[2] * 40000];
    const sunModelView = mat4.clone(this.viewMatrix);
    mat4.translate(sunModelView, sunModelView, sunWorldPos as any);
    
    // Cancel rotation (billboarding)
    sunModelView[0] = 10000; sunModelView[1] = 0; sunModelView[2] = 0;
    sunModelView[4] = 0; sunModelView[5] = 10000; sunModelView[6] = 0;
    sunModelView[8] = 0; sunModelView[9] = 0; sunModelView[10] = 10000;

    gl.uniformMatrix4fv(this.uModelView, false, sunModelView);
    gl.uniform1f(this.uIsEarth, 0.2); // Sun mode
    gl.disable(gl.DEPTH_TEST);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // Additive blend for sun
    gl.bindVertexArray(this.vaoSat); // Reuse sphere VAO but it will look like a glowy blob
    gl.drawElements(gl.TRIANGLES, this.numSatIndices, gl.UNSIGNED_SHORT, 0);
    gl.enable(gl.DEPTH_TEST);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // 4. Telemetry Trails
    gl.uniform1f(this.uIsEarth, 0.0);
    for (let i = 0; i < 4; i++) {
      if (!this.activeSats[i] || this.trailCounts[i] < 2) continue;

      gl.uniform3fv(this.uColor, this.satColors[i]);
      gl.bindVertexArray(this.trailVAOs[i]);
      
      if (this.trailCounts[i] < 200) {
        gl.drawArrays(gl.LINE_STRIP, 0, this.trailCounts[i]);
      } else {
        gl.drawArrays(gl.LINE_STRIP, this.trailHeads[i], 200 - this.trailHeads[i]);
        gl.drawArrays(gl.LINE_STRIP, 0, this.trailHeads[i]);
      }
    }

    // 5. Target Orbit Ring
    {
      const identityMV = mat4.clone(this.viewMatrix);
      gl.uniformMatrix4fv(this.uModelView, false, identityMV);
      gl.uniform1f(this.uIsEarth, 0.0);
      gl.uniform3f(this.uColor, 0.0, 0.78, 1.0); // Cyan
      gl.enable(gl.BLEND);
      gl.bindVertexArray(this.targetOrbitVAO);
      gl.drawArrays(gl.LINE_LOOP, 0, WebGLContext.ORBIT_SEGMENTS);
      gl.disable(gl.BLEND);
    }
  }

  dispose(): void {}
}
