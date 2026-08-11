(function (root) {
    "use strict";

    const portableLiteral = /^[a-z0-9][a-z0-9._-]*$/;
    const encodedComponent = /^~(?:[0-9a-f]{2})+$/;
    const windowsReservedStems = new Set([
        "aux", "con", "nul", "prn",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    ]);

    function isPortableLiteral(value) {
        if (!portableLiteral.test(value) || value.endsWith(".")) {
            return false;
        }
        return !windowsReservedStems.has(value.split(".", 1)[0]);
    }

    function toPathComponent(modelId) {
        if (typeof modelId !== "string") {
            throw new TypeError("modelId must be a string");
        }
        if (modelId.length === 0) {
            throw new Error("modelId must not be empty");
        }
        if (isPortableLiteral(modelId)) {
            return modelId;
        }

        const bytes = new TextEncoder().encode(modelId);
        return `~${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
    }

    function fromPathComponent(component) {
        if (typeof component !== "string") {
            throw new TypeError("component must be a string");
        }
        if (component.length === 0) {
            throw new Error("component must not be empty");
        }
        if (isPortableLiteral(component)) {
            return component;
        }
        if (!encodedComponent.test(component)) {
            throw new Error("component is not a canonical model path component");
        }

        const pairs = component.slice(1).match(/.{2}/g);
        const bytes = Uint8Array.from(pairs, (pair) => Number.parseInt(pair, 16));
        const modelId = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
        if (toPathComponent(modelId) !== component) {
            throw new Error("component is a noncanonical model path alias");
        }
        return modelId;
    }

    const api = Object.freeze({
        fromPathComponent,
        toPathComponent,
    });

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.EconBenchModelIds = api;
}(typeof globalThis !== "undefined" ? globalThis : this));
