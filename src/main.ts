import {vec3, vec2, mat4} from 'gl-matrix';
import * as Stats from 'stats-js';
import * as DAT from 'dat-gui';
import Square from './geometry/Square';
import Camera from './Camera';
import {setGL} from './globals';
import ShaderProgram, {Shader} from './rendering/gl/ShaderProgram';
//import { inverse } from 'gl-matrix';

// Define an object with application parameters and button callbacks
// This will be referred to by dat.GUI's functions that add GUI elements.
const controls = {
  // TODO: add any controls you want
  colorScheme: 0.0,
  pause: false,
  reflective: false,
};

let screenQuad: Square;
let time: number = 0;

function main() {
  // Initial display for framerate
  const stats = Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0px';
  stats.domElement.style.top = '0px';
  document.body.appendChild(stats.domElement);

  // TODO: add any controls you need to the gui
  const gui = new DAT.GUI();
  // E.G. gui.add(controls, 'tesselations', 0, 8).step(1);
  // Choose from accepted values
  gui.add(controls, 'colorScheme', { 'Scheme1': 0.0, 'Scheme2': 1.0, 'Scheme3': 2.0 } );
  gui.add(controls, 'pause', false);
  gui.add(controls, 'reflective', false);

  // get canvas and webgl context
  const canvas = <HTMLCanvasElement> document.getElementById('canvas');

  function setSize(width: number, height: number) {
    canvas.width = width;
    canvas.height = height;
  }

  const gl = <WebGL2RenderingContext> canvas.getContext('webgl2');
  if (!gl) {
    alert('WebGL 2 not supported!');
  }


  // `setGL` is a function imported above which sets the value of `gl` in the `globals.ts` module.
  // Later, we can import `gl` from `globals.ts` to access it
  setGL(gl);

  screenQuad = new Square(vec3.fromValues(0, 0, 0));
  screenQuad.create();

  const camera = new Camera(vec3.fromValues(0, 0, 5), vec3.fromValues(0, 0, 0));

  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.disable(gl.DEPTH_TEST);


  const raymarchShader = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/screenspace-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/raymarch-frag.glsl')),
  ]);


  // This function will be called every frame
  function tick() {
    camera.update();
    stats.begin();

    raymarchShader.setTime(time);
    time++;
    raymarchShader.setScheme(controls.colorScheme.valueOf());
    if(controls.pause.valueOf() == true) {
      time = 0;
    }
    if(controls.reflective == false) {
      raymarchShader.setReflect(0.0);
    } else {
      raymarchShader.setReflect(1.0);
    }

    gl.viewport(0, 0, window.innerWidth, window.innerHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // TODO: get / calculate relevant uniforms to send to shader here
    // TODO: send uniforms to shader
    raymarchShader.setDimensions(vec2.fromValues(window.innerWidth, window.innerHeight));
    // March!
    raymarchShader.draw(screenQuad);

    // TODO: more shaders to layer / process the first one? (either via framebuffers or blending)

    stats.end();

    // Tell the browser to call `tick` again whenever it renders a new frame
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', function() {
    setSize(window.innerWidth, window.innerHeight);
    camera.setAspectRatio(window.innerWidth / window.innerHeight);
    camera.updateProjectionMatrix();
  }, false);

  setSize(window.innerWidth, window.innerHeight);
  camera.setAspectRatio(window.innerWidth / window.innerHeight);
  camera.updateProjectionMatrix();
  raymarchShader.setDimensions(vec2.fromValues(window.innerWidth, window.innerHeight));

  // Start the render loop
  tick();
}

main();
