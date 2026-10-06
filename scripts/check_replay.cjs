// Developer-only DOM logic test for our generated offline replay; no browser/user UI control.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.handlers = {}; this.style = {}; this.value = "850"; this._text = ""; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this._text = ""; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, action) { this.handlers[name] = action; }
  click() { assert.equal(typeof this.handlers.click, "function"); this.handlers.click(); }
}

const filename = process.argv[2];
assert.ok(filename, "Pass the generated demo_replay.html path");
const html = fs.readFileSync(filename, "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1);
assert.ok(!scripts[0][1].includes("__GENREC_REPLAY_DATA__"));
const nodes = new Map();
const get = id => { if (!nodes.has(id)) nodes.set(id, new Element("div")); return nodes.get(id); };
const pending = new Map();
let timerId = 0;
const context = vm.createContext({
  document: { getElementById: get, createElement: tag => new Element(tag), querySelectorAll: () => get("tabs").children },
  setTimeout: callback => { pending.set(++timerId, callback); return timerId; },
  clearTimeout: id => pending.delete(id),
});
vm.runInContext(scripts[0][1], context, { timeout: 1000 });
assert.equal(get("tabs").children.length, 3);
assert.match(get("notice").textContent, /4\/12/);
assert.match(get("notice").textContent, /1\/9/);
const outcomes = [];
for (const [tab, steps, products] of [[0, 7, 5], [1, 6, 5], [2, 2, 0]]) {
  get("tabs").children[tab].click();
  assert.equal(get("counter").textContent, `0 / ${steps}`);
  assert.equal(get("products").children.filter(row => row.className === "product").length, 0);
  get("next").click();
  assert.equal(get("counter").textContent, `1 / ${steps}`);
  get("play").click();
  get("play").click(); // pause must cancel the queued animation
  assert.equal(pending.size, 0);
  get("play").click();
  let iterations = 0;
  while (pending.size) {
    assert.ok(++iterations < 20);
    const [id, action] = pending.entries().next().value;
    pending.delete(id);
    action();
  }
  assert.equal(get("counter").textContent, `${steps} / ${steps}`);
  assert.equal(get("products").children.filter(row => row.className === "product").length, products);
  if (tab === 2) assert.match(get("outcome").textContent, /澄清/);
  get("reset").click();
  assert.equal(get("counter").textContent, `0 / ${steps}`);
  outcomes.push({ tab, steps, products, pause_and_reset: "passed" });
}
console.log(JSON.stringify({ scope: "offline Javascript DOM logic; not a benchmark or live browser interaction", outcomes }, null, 2));
