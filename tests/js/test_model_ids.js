"use strict";

const assert = require("node:assert/strict");
const modelIds = require("../../web/model_ids.js");

const vectors = new Map([
    ["gpt-4o", "gpt-4o"],
    ["gpt-5.2", "gpt-5.2"],
    ["a/b", "~612f62"],
    ["a:b", "~613a62"],
    ["a_b", "a_b"],
    ["A", "~41"],
    [".", "~2e"],
    ["con", "~636f6e"],
    ["con.txt", "~636f6e2e747874"],
    ["model.", "~6d6f64656c2e"],
    ["模型/甲", "~e6a8a1e59e8b2fe794b2"],
]);

for (const [modelId, expected] of vectors) {
    const component = modelIds.toPathComponent(modelId);
    assert.equal(component, expected);
    assert.equal(modelIds.fromPathComponent(component), modelId);
}

for (const component of ["", "A", "a/b", "~", "~0", "~zz", "~ff", "~677074"]) {
    assert.throws(() => modelIds.fromPathComponent(component));
}

assert.throws(() => modelIds.toPathComponent(""));
assert.throws(() => modelIds.toPathComponent(null));

const formerlyColliding = ["a/b", "a:b", "a_b"].map(modelIds.toPathComponent);
assert.equal(new Set(formerlyColliding).size, formerlyColliding.length);
