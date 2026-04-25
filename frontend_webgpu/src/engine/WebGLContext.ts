import { mat4, vec3 } from 'gl-matrix';
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
out vec2 vUV;
out vec3 vPosition;

void main() {
  vec4 pos = uModelViewMatrix * vec4(aVertexPosition, 1.0);
  gl_Position = uProjectionMatrix * pos;
  vPosition = pos.xyz;
  vNormal = mat3(uNormalMatrix) * aNormal;
  vUV = aUV;
  vColor = uColor;
}
`;

const fsSource = `#version 300 es
precision highp float;

in vec3 vColor;
in vec3 vNormal;
in vec2 vUV;
in vec3 vPosition;

uniform vec3 uSunDir;
uniform sampler2D uHeatmap;
uniform float uIsEarth; // 1.0 = solid, 0.5 = wireframe, 0.0 = sat
uniform float uAtmDensity;

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
    float diff = max(dot(N, L), 0.0);
    
    // Earth surface color (Ocean blue)
    vec3 earthBase = vec3(0.1, 0.2, 0.5);
    vec3 litColor = earthBase * (0.4 + 0.6 * diff); // Increased ambient from 0.2 to 0.4
    
    // Sample heatmap
    float heat = texture(uHeatmap, vec2(vUV.x, 1.0 - vUV.y)).r;
    vec3 heatColor = vec3(0.0);
    if (heat > 0.01) {
        heatColor = colormap(heat) * 1.5;
        // Optional: Contour lines
        float contour = smoothstep(0.05, 0.0, abs(fract(heat * 10.0 + 0.5) - 0.5));
        heatColor += vec3(contour * 0.3);
    }
    
    // Atmospheric Glow
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - max(dot(N, viewDir), 0.0), 3.0);
    vec3 atmColor = vec3(0.4, 0.7, 1.0) * fresnel * (0.5 + uAtmDensity * 3.0);
    
    // Mix earth and heatmap
    vec3 finalColor = litColor;
    if (heat > 0.1) {
        finalColor = mix(litColor, heatColor, heat * 0.8);
    } else {
        finalColor += heatColor * 0.5;
    }
    
    fragColor = vec4(finalColor + atmColor, 1.0);
  } else if (uIsEarth > 0.2) {
    fragColor = vec4(vColor, 0.2); // Wireframe
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
  private vaoSat!: WebGLVertexArrayObject;
  private numSatIndices = 0;

  private projectionMatrix = mat4.create();
  private viewMatrix = mat4.create();
  private satPosition = vec3.fromValues(0, 0, 0);
  private satColor = vec3.fromValues(0.2, 0.5, 1.0);
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
  
  private heatmapTex!: WebGLTexture;

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

    const earthGeo = createSphere(6371, 32, 32);
    this.vaoEarth = this.createVao(earthGeo.positions, earthGeo.normals, earthGeo.uvs, earthGeo.indices);
    this.numEarthIndices = earthGeo.indices.length;

    const satGeo = createSphere(100, 8, 8);
    this.vaoSat = this.createVao(satGeo.positions, satGeo.normals, satGeo.uvs, satGeo.indices);
    this.numSatIndices = satGeo.indices.length;

    this.loadHeatmap();

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
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
    const gl = this.gl;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.01, 0.01, 0.02, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.program);

    const aspect = gl.canvas.width / gl.canvas.height;
    mat4.perspective(this.projectionMatrix, 45 * Math.PI / 180, aspect, 100.0, 50000.0);
    gl.uniformMatrix4fv(this.uProj, false, this.projectionMatrix);

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
    mat4.rotateY(earthModelView, earthModelView, time * 0.00005);
    gl.uniformMatrix4fv(this.uModelView, false, earthModelView);
    
    const normalMatrix = mat4.create();
    mat4.invert(normalMatrix, earthModelView);
    mat4.transpose(normalMatrix, normalMatrix);
    gl.uniformMatrix4fv(this.uNormalMatrix, false, normalMatrix);

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

    // 2. Sat
    const satModelView = mat4.clone(this.viewMatrix);
    mat4.translate(satModelView, satModelView, this.satPosition);
    gl.uniformMatrix4fv(this.uModelView, false, satModelView);
    gl.uniform1f(this.uIsEarth, 0.0);
    gl.uniform3fv(this.uColor, this.satColor);

    gl.bindVertexArray(this.vaoSat);
    gl.enable(gl.CULL_FACE);
    gl.drawElements(gl.TRIANGLES, this.numSatIndices, gl.UNSIGNED_SHORT, 0);
  }

  dispose(): void {}
}
