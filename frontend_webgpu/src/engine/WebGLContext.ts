import { mat4, vec3 } from 'gl-matrix';
import type { IRenderContext } from './Renderer';
import type { TelemetryData } from '../lib/telemetry';
import { createSphere } from './geometry';

const vsSource = `#version 300 es
layout(location = 0) in vec3 aVertexPosition;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec3 uColor;

out vec3 vColor;

void main() {
  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);
  vColor = uColor;
  
  // Basic depth fading
  float depth = (gl_Position.z / gl_Position.w + 1.0) * 0.5;
  vColor = uColor * (1.2 - depth);
}
`;

const fsSource = `#version 300 es
precision highp float;

in vec3 vColor;
out vec4 fragColor;

void main() {
  fragColor = vec4(vColor, 1.0);
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

  // Uniform locations
  private uProj!: WebGLUniformLocation;
  private uModelView!: WebGLUniformLocation;
  private uColor!: WebGLUniformLocation;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('WebGL2 not supported');
    this.gl = gl;

    // Compile Shaders
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
    this.uColor = gl.getUniformLocation(this.program, 'uColor')!;

    // Setup Earth (Wireframe sphere)
    const earthGeo = createSphere(6371, 32, 32);
    this.vaoEarth = this.createVao(earthGeo.positions, earthGeo.indices);
    this.numEarthIndices = earthGeo.indices.length;

    // Setup Satellite (Small solid sphere)
    const satGeo = createSphere(100, 8, 8); // Scaled up slightly for visibility
    this.vaoSat = this.createVao(satGeo.positions, satGeo.indices);
    this.numSatIndices = satGeo.indices.length;

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
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

  private createVao(positions: Float32Array, indices: Uint16Array): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    return vao;
  }

  updateTelemetry(data: TelemetryData): void {
    // ECI to ECEF roughly (ignoring exact time rotation for visual simplicity)
    const latRad = data.latitudeDeg * Math.PI / 180;
    const lonRad = data.longitudeDeg * Math.PI / 180;
    const r = 6371 + data.altitudeKm;

    const x = r * Math.cos(latRad) * Math.cos(lonRad);
    const y = r * Math.cos(latRad) * Math.sin(lonRad);
    const z = r * Math.sin(latRad);

    // Swap Y/Z to match WebGL/WebGPU common coords (Y-up)
    this.satPosition[0] = x;
    this.satPosition[1] = z;
    this.satPosition[2] = -y;

    // Color based on FDIR
    if (data.fdirMode === 2) { // Safe (Red)
      vec3.set(this.satColor, 1.0, 0.2, 0.2);
    } else if (data.fdirMode === 1) { // Degraded (Amber)
      vec3.set(this.satColor, 1.0, 0.7, 0.1);
    } else if (data.payloadOn) { // Payload ON (Green)
      vec3.set(this.satColor, 0.1, 1.0, 0.4);
    } else { // Nominal (Blue)
      vec3.set(this.satColor, 0.2, 0.5, 1.0);
    }
  }

  render(time: number): void {
    const gl = this.gl;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.02, 0.02, 0.04, 1.0); // Deep space
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.program);

    // Camera setup
    const aspect = gl.canvas.width / gl.canvas.height;
    mat4.perspective(this.projectionMatrix, 45 * Math.PI / 180, aspect, 100.0, 30000.0);
    gl.uniformMatrix4fv(this.uProj, false, this.projectionMatrix);

    // Slowly rotate camera around earth
    const camAngle = time * 0.0001;
    const camRadius = 15000;
    mat4.lookAt(this.viewMatrix, 
      [Math.cos(camAngle) * camRadius, 4000, Math.sin(camAngle) * camRadius],
      [0, 0, 0],
      [0, 1, 0]
    );

    // 1. Draw Earth (Wireframe effect using LINE_STRIP or points, but we'll use triangles with blending or just solid dark)
    const earthModelView = mat4.clone(this.viewMatrix);
    // Rotate earth slowly
    mat4.rotateY(earthModelView, earthModelView, time * 0.00005);
    gl.uniformMatrix4fv(this.uModelView, false, earthModelView);
    gl.uniform3f(this.uColor, 0.15, 0.35, 0.6); // Brighter cyan-blue wireframe

    gl.bindVertexArray(this.vaoEarth);
    // Use line loop for cool wireframe aesthetic
    gl.drawElements(gl.LINES, this.numEarthIndices, gl.UNSIGNED_SHORT, 0);

    // 2. Draw Satellite
    const satModelView = mat4.clone(this.viewMatrix);
    mat4.translate(satModelView, satModelView, this.satPosition);
    gl.uniformMatrix4fv(this.uModelView, false, satModelView);
    gl.uniform3fv(this.uColor, this.satColor);

    gl.bindVertexArray(this.vaoSat);
    gl.drawElements(gl.TRIANGLES, this.numSatIndices, gl.UNSIGNED_SHORT, 0);
  }

  dispose(): void {
    // Basic cleanup
  }
}
