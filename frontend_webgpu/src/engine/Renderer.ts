import type { TelemetryData } from '../lib/telemetry';
import { WebGPUContext } from './WebGPUContext';
import { WebGLContext } from './WebGLContext';

export interface IRenderContext {
  init(canvas: HTMLCanvasElement): Promise<void>;
  updateTelemetry(data: TelemetryData): void;
  render(time: number): void;
  dispose(): void;
}

export default class Renderer {
  private canvas: HTMLCanvasElement;
  private context: IRenderContext | null = null;
  private animationFrameId: number = 0;
  private disposed: boolean = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async init() {
    // 1. Resize canvas to screen
    this.resize();
    window.addEventListener('resize', this.resize.bind(this));

    // 2. Check for WebGPU support
    if (navigator.gpu) {
      console.log('WebGPU is supported. Initializing WebGPU Context...');
      try {
        this.context = new WebGPUContext();
        await this.context.init(this.canvas);
      } catch (e) {
        console.warn('WebGPU init failed, falling back to WebGL2', e);
        this.initWebGL();
      }
    } else {
      console.log('WebGPU NOT supported. Falling back to WebGL2 Context...');
      this.initWebGL();
    }

    // 3. Start render loop
    this.renderLoop = this.renderLoop.bind(this);
    this.animationFrameId = requestAnimationFrame(this.renderLoop);
  }

  private initWebGL() {
    this.context = new WebGLContext();
    this.context.init(this.canvas);
  }

  private resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  public updateTelemetry(data: TelemetryData) {
    if (this.context) {
      this.context.updateTelemetry(data);
    }
  }

  private renderLoop(time: number) {
    if (this.disposed) return;
    
    if (this.context) {
      this.context.render(time);
    }
    
    this.animationFrameId = requestAnimationFrame(this.renderLoop);
  }

  public dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animationFrameId);
    window.removeEventListener('resize', this.resize.bind(this));
    if (this.context) {
      this.context.dispose();
    }
  }
}
