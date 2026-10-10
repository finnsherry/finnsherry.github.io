export class InputState {
    // Copied from https://bellaard.com/js/inputstate.js
    constructor() {
        this.pointers = new Map();
        this.pointersMovementX = new Map();
        this.pointersMovementY = new Map();
        this.toBeFlushed = new Set();

        this.pointerWentDown = new Set();
        this.pointerWentUp = new Set();

        this.wheelTicks = { x: 0, y: 0 };

        this.keyCodeDown = {};
        this.keyCodeWentDown = new Set();
        this.keyCodeWentUp = new Set();
    }

    // Attach event listeners to an element to track input state
    // this will prevent default behavior for most events, so it should be used on a dedicated input element (e.g. a canvas)
    attach(element) {
        element.tabIndex = 0; // Make the element focusable

        element.style.touchAction = "none"; // disable default touch actions


        element.addEventListener("dragstart", e => {
            e.preventDefault();
        });

        element.addEventListener("contextmenu", e => {
            e.preventDefault()
        });

        element.addEventListener("pointerdown", e => {
            // console.log("pointerdown", e);
            element.setPointerCapture(e.pointerId);
            this.pointers.set(e.pointerId, e);
            this.pointersMovementX.set(e.pointerId, 0);
            this.pointersMovementY.set(e.pointerId, 0);
            this.pointerWentDown.add(e.pointerId);
        });

        element.addEventListener("pointermove", e => {
            // console.log("pointermove", e);
            // Update the pointer event with the latest position, but accumulate properties such as movementX/Y
            // unfortunately, e.movementX/Y can not be set (read-only)
            // also, we can not rely on the browser to provide movementX/Y, because it can be inconsistent?
            // so we will calculate movementX/Y ourselves by comparing the current event with the previous event for the same pointerId

            const prev = this.pointers.get(e.pointerId);

            const movementX = (prev ? e.clientX - prev.clientX : 0) + this.pointersMovementX.get(e.pointerId) || 0;
            const movementY = (prev ? e.clientY - prev.clientY : 0) + this.pointersMovementY.get(e.pointerId) || 0;

            this.pointers.set(e.pointerId, e);
            this.pointersMovementX.set(e.pointerId, movementX);
            this.pointersMovementY.set(e.pointerId, movementY);
        });

        element.addEventListener("pointerup", e => {
            // console.log("pointerup", e);
            this.pointerWentUp.add(e.pointerId);
            this.toBeFlushed.add(e.pointerId);
        });

        element.addEventListener("pointercancel", e => {
            // console.log("pointercancel", e);
            this.toBeFlushed.add(e.pointerId);
        });

        element.addEventListener("wheel", e => {
            e.preventDefault();
            this.wheelTicks.x += Math.sign(e.deltaX);
            this.wheelTicks.y += Math.sign(e.deltaY);
        });

        element.addEventListener("keydown", e => {
            // console.log("keydown", e);
            e.preventDefault();
            this.keyCodeDown[e.code] = true;
            this.keyCodeWentDown.add(e.code);
        });

        element.addEventListener("keyup", e => {
            // console.log("keyup", e);
            this.keyCodeDown[e.code] = false;
            this.keyCodeWentUp.add(e.code);
        });
    }

    flush() {
        // clear accumulated movementX/Y for all pointers
        this.pointersMovementX.forEach((_, key) => this.pointersMovementX.set(key, 0));
        this.pointersMovementY.forEach((_, key) => this.pointersMovementY.set(key, 0));
        // clear wheel ticks
        this.wheelTicks = { x: 0, y: 0 };
        // clear keyCodeWentDown and Up
        this.keyCodeWentDown.clear();
        this.keyCodeWentUp.clear();
        // clear pointerWentDown and Up
        this.pointerWentDown.clear();
        this.pointerWentUp.clear();
        // remove pointers that went up or were canceled
        this.toBeFlushed.forEach(pointerId => {
            this.pointers.delete(pointerId);
            this.pointersMovementX.delete(pointerId);
            this.pointersMovementY.delete(pointerId);
        });
        this.toBeFlushed.clear();
    }
}


export class InputForm {
    constructor(container) {
        this.container = container;
        this.table = document.createElement("table");
        this.container.appendChild(this.table);
        this.table.classList.add('form');

        this.defaults = new Map();
    }

    addThing(shownLabel) {
        const tr = document.createElement("tr");
        const tdShownLabel = document.createElement("td");
        const htmlShownLabel = document.createElement("label");
        htmlShownLabel.textContent = shownLabel + ":";
        tdShownLabel.append(htmlShownLabel);
        tr.appendChild(tdShownLabel);
        return tr
    }

    addNumber({ label, shownLabel, defaultValue, width = null }) {
        const tr = this.addThing(shownLabel);
        const tdNumber = document.createElement("td");
        const input = document.createElement("input");
        input.type = "text";
        input.id = label;
        input.name = label;
        input.value = defaultValue;
        input.required = true;
        if (width) {
            input.size = width;
        }
        tdNumber.append(input);
        tr.appendChild(tdNumber);

        this.table.appendChild(tr);
        this.defaults.set(label, defaultValue);
        return input
    }

    readNumber(label) {
        const entry = document.getElementById(label);
        return entry.value
    }

    writeNumber(label, value) {
        const entry = document.getElementById(label);
        if (Number.isFinite(value)) {
            entry.value = value;
        }
    }

    addVector({ label, shownLabel, defaultValue, width = null }) {
        const tr = this.addThing(shownLabel);
        const tdVector = document.createElement("td");
        let out = [];
        label.forEach((l, i) => {
            const tdEntry = document.createElement("td");
            const input = document.createElement("input");
            input.type = "text";
            input.id = l;
            input.name = l;
            input.value = defaultValue[i];
            input.required = true;
            if (width) {
                input.size = width;
            }
            tdEntry.append(input);
            tdVector.appendChild(tdEntry);
            this.defaults.set(l, defaultValue[i]);
            out.push(defaultValue)
        });
        tr.append(tdVector);

        this.table.appendChild(tr);
        return out
    }

    readVector(labels) {
        return labels.map((l) => document.getElementById(l).value);
    }

    writeVector(labels, values) {
        return labels.forEach((l, i) => {
            const entry = document.getElementById(l);
            if (Number.isFinite(values[i])) {
                entry.value = values[i];
            }
        })
    }

    addRange({ label, shownLabel, defaultValue, min, max, width = null }) {
        const tr = this.addThing(shownLabel);
        const tdSlider = document.createElement("td");
        const range = document.createElement("input");
        range.type = "range";
        range.id = label;
        range.name = shownLabel;
        range.min = min;
        range.max = max;
        range.step = "any";
        range.value = defaultValue;
        if (width) {
            range.size = width;
        }
        tdSlider.append(range);
        tr.appendChild(tdSlider);

        const tdShownValue = document.createElement("td");
        let htmlShownValue = document.createElement("label");
        htmlShownValue.textContent = defaultValue.toFixed(3);
        tdShownValue.append(htmlShownValue);

        range.addEventListener('input', () => {
            htmlShownValue.textContent = Number(range.value).toFixed(3);
        })
        tr.appendChild(tdShownValue);


        this.table.appendChild(tr);
        this.defaults.set(label, defaultValue);
        return range
    }

    readRange(label) {
        const entry = document.getElementById(label);
        return parseFloat(entry.value);
    }

    writeRange(label, value) {
        const entry = document.getElementById(label);
        if (Number.isFinite(value)) {
            entry.value = value;
            entry.dispatchEvent(new Event("input"));
        }
    }

    addSelector({ label, shownLabel, defaultValue, options, width = null }) {
        const tr = this.addThing(shownLabel);
        const tdSelector = document.createElement("td");
        const selector = document.createElement("select");
        selector.id = label;
        if (width) {
            selector.width = width;
        }

        for (const [value, shownValue] of options) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = shownValue;
            selector.appendChild(option);
        }
        tdSelector.appendChild(selector);
        tr.append(tdSelector);

        this.table.appendChild(tr);
        this.defaults.set(label, defaultValue);
        return selector
    }

    readSelector(label) {
        const entry = document.getElementById(label);
        return entry.value
    }

    writeSelector(label, value) {
        const entry = document.getElementById(label);
        entry.value = value;
    }

    getThing(label) {
        return document.getElementById(label)
    }

    reset() {
        for (const [label, defaultValue] of this.defaults) {
            const entry = document.getElementById(label);
            entry.value = defaultValue;
        }

    }

    remove(label) {
        document.getElementById(label).closest("tr").remove();
        entry.remove();
    }
}


export class InputButtons {
    constructor(container, table = null) {
        this.container = container;
        if (table) {
            this.table = table;
        } else {
            this.table = document.createElement("table");
            this.container.appendChild(this.table);
        }

        const tr = document.createElement("tr");
        this.table.appendChild(tr);
        this.tr = tr;
    }

    addButton({ label, shownLabel, width = null }) {
        const td = document.createElement("td");
        const button = document.createElement("button");
        button.id = label;
        button.name = label;
        button.textContent = shownLabel;
        if (width) {
            button.size = width;
        }
        td.append(button);
        this.tr.appendChild(td);
        return button
    }

    getButton(label) {
        return document.getElementById(label)
    }

    remove(label) {
        document.getElementById(label).closest("tr").remove();
    }
}