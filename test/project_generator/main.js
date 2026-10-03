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

let running = false;

function randomWord(options) {
    return options[Math.floor(Math.random() * options.length)];
}

function spin() {
    if (running) return;
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
        }, 80);

        const stopDelay = 1800 + index * 1100;

        setTimeout(() => {
            clearInterval(interval);
            wheel.classList.remove("spinning");
            span.textContent = randomWord(wordList);

            stopped++;

            if (stopped === results) {
                running = false;
                spinButton.disabled = false;
            }
        }, stopDelay);
    });
}

spinButton.addEventListener("click", spin)