import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.163/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.163/examples/jsm/controls/OrbitControls.js';

const form = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');
const maskPreview = document.getElementById('maskPreview');
const depthPreview = document.getElementById('depthPreview');

const canvas = document.getElementById('threeCanvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020617);

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
camera.position.set(0, 0.25, 2.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0.6);

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, 2, 2);
scene.add(dirLight);

let cloudObject = null;

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}

window.addEventListener('resize', resize);
resize();

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

function setStatus(text) {
  statusEl.textContent = text;
}

function clearCloud() {
  if (!cloudObject) return;
  scene.remove(cloudObject);
  cloudObject.geometry.dispose();
  cloudObject.material.dispose();
  cloudObject = null;
}

function renderCloud(payload) {
  clearCloud();

  const positions = new Float32Array(payload.points.flat());
  const colors = new Float32Array(payload.colors.flat());

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    size: 0.01,
    vertexColors: true,
    sizeAttenuation: true,
  });

  cloudObject = new THREE.Points(geometry, material);
  scene.add(cloudObject);

  if (geometry.boundingSphere) {
    const center = geometry.boundingSphere.center;
    controls.target.copy(center);
    camera.position.set(center.x, center.y + 0.25, center.z + geometry.boundingSphere.radius * 2.8);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = document.getElementById('imageInput');
  if (!input.files?.length) {
    setStatus('Choose an image first.');
    return;
  }

  const data = new FormData();
  data.append('image', input.files[0]);

  setStatus('Running segmentation + depth estimation...');
  metaEl.innerHTML = '';

  try {
    const response = await fetch('/api/reconstruct', { method: 'POST', body: data });
    if (!response.ok) {
      throw new Error(`Server error ${response.status}`);
    }

    const result = await response.json();
    const cloud = await fetch(result.pointCloudUrl).then((r) => r.json());
    renderCloud(cloud);

    maskPreview.src = result.maskUrl;
    depthPreview.src = result.depthUrl;

    metaEl.innerHTML = [
      `<li>Subject Label: ${result.label}</li>`,
      `<li>Segmentation Confidence: ${Number(result.confidence).toFixed(3)}</li>`,
      `<li>3D Points Generated: ${result.pointCount.toLocaleString()}</li>`,
    ].join('');

    setStatus('Complete. Drag to inspect the reconstructed object in 3D.');
  } catch (error) {
    console.error(error);
    setStatus(`Failed: ${error.message}`);
  }
});
