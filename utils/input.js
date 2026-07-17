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


export class InputTable {
    constructor(container) {
        this.container = container;
        const table = document.createElement("table");
        table.classList.add('form');
        this.container.appendChild(table);
        this.table = table;

        this.numbers = new Map();
        this.selectors = new Map();
    }

    addNumber({ label, shownLabel, defaultValue, width = null }) {
        const tr = document.createElement("tr");
        const tdShownLabel = document.createElement("td");
        const htmlShownLabel = document.createElement("label");
        htmlShownLabel.textContent = shownLabel + ":";
        tdShownLabel.append(htmlShownLabel);
        tr.appendChild(tdShownLabel);
        switch (typeof (label)) {
            case "string":
                // single column
                this.addNumberColumn(tr, label, defaultValue, width);
                break;
            case "object":
                // multiple columns as a "vector"
                label.forEach((l, i) => this.addNumberColumn(tr, l, defaultValue[i], width));
                break;
        }
        this.table.appendChild(tr);
    }

    addNumberColumn(tr, label, defaultValue, width = null) {
        const td = document.createElement("td");
        const input = document.createElement("input");
        input.type = "text";
        input.id = label;
        input.name = label;
        input.value = defaultValue;
        input.required = true;
        if (width) {
            input.size = width;
        }
        td.append(input);
        tr.appendChild(td);
        this.numbers.set(label, { value: defaultValue, defaultValue: defaultValue });
    }

    getNumber(label, isInt = false) {
        const entry = document.getElementById(label);
        const param = this.numbers.get(label);
        let value = entry["value"];
        if (value === '' || value === null || value === undefined) {
            value = param.get("defaultValue");
        } else {
            if (isInt) {
                value = parseInt(value);
            } else {
                value = parseFloat(value);
            }
            param.value = value;
        }
        return value;
    }

    setNumber(label, value) {
        const param = this.numbers.get(label);
        param.value = value;
        const entry = document.getElementById(label);
        entry.value = value;
    }

    addSelector({ label, shownLabel, options, width = null }) {
        const tr = document.createElement("tr");
        const tdShownLabel = document.createElement("td");
        const htmlShownLabel = document.createElement("label");
        htmlShownLabel.textContent = shownLabel + ":";
        tdShownLabel.append(htmlShownLabel);
        tr.appendChild(tdShownLabel);

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
        this.selectors.set(label, options[0][0]);
    }

    getSelector(label) {
        const entry = document.getElementById(label);
        const value = entry.value;
        this.selectors.set(label, value);
        return value;
    }

    setSelector(label, value) {
        this.selectors.set(label, value);
        const entry = document.getElementById(label);
        entry.value = value;
    }
}


export class InputButtons {
    constructor(container) {
        this.container = container;
        const table = document.createElement("table");
        table.classList.add('form');
        this.container.appendChild(table);
        this.table = table;

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
    }
}