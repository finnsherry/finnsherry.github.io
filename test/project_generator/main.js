import { InputButtons } from "/utils/input.js";


const container = document.getElementsByClassName("machine")[0];
const inputButtons = new InputButtons(container);
const spinButton = inputButtons.addButton({ label: "spin", shownLabel: "SPIN" });
spinButton.classList.add("spin-button")
console.log(spinButton);

const groups = [
    "SE(2)", "SE(3)", "SO(3)", "SIM(2)",
];
const symmetries = [
    "Equivariant", "Data-driven",
];
const applications = [
    "Tracking", "Denoising", "Segmentation", "Inpainting",
    "Classification", "Generative Modelling",
];
let wordLists = [groups, symmetries, applications];

const wheels = document.querySelectorAll(".wheel");
const results = wheels.length;

let audioCtx;
let soundTimer;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
}

function clickSound() {
    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "square";
    osc.frequency.setValueAtTime(1200 + Math.random() * 1000, now);
    osc.frequency.exponentialRampToValueAtTime(180, now + 0.035);

    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(now);
    osc.stop(now + 0.045);
}

function stopSound() {
    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(900, now);
    osc.frequency.exponentialRampToValueAtTime(90, now + 0.4);

    gain.gain.setValueAtTime(0.3, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(now);
    osc.stop(now + 0.4);
}

function jackpotSound() {
    const notes = [523, 659, 784, 1047, 784, 1047, 1568];

    notes.forEach((freq, i) => {
        const now = audioCtx.currentTime + i * 0.16;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = "square";
        osc.frequency.value = freq;

        gain.gain.setValueAtTime(0.25, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.35);
    });
}

let running = false;

function randomWord(options) {
    return options[Math.floor(Math.random() * options.length)];
}

function spin() {
    if (running) return;
    initAudio();
    running = true;
    spinButton.disabled = true;

    let stopped = 0;
    const WEEDEATER = Math.random() < 0.01;
    if (WEEDEATER) {
        wordLists = [["WEED"], ["EATER"], ["🌿"]];
    }

    wheels.forEach((wheel, index) => {
        const span = wheel.querySelector("span");
        wheel.classList.add("spinning");
        const wordList = wordLists[index]

        const interval = setInterval(() => {
            span.textContent = randomWord(wordList);
            clickSound();
        }, 80);

        const stopDelay = 1800 + index * 1100;

        setTimeout(() => {
            clearInterval(interval);
            wheel.classList.remove("spinning");
            stopSound();
            span.textContent = randomWord(wordList);

            stopped++;

            if (stopped === results) {
                running = false;
                spinButton.disabled = false;
                jackpotSound();
            }
        }, stopDelay);
    });
}

spinButton.addEventListener("click", spin)