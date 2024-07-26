window.onload = function init() {
    const decisionTree = document.getElementById("decisionTree");
    if (decisionTree) {
        svg = decisionTree.contentDocument.documentElement;
        svg.setAttribute("id", "innerSvg");
        svg.addEventListener('wheel', zoom, { passive: false });
        svg.addEventListener('mousedown', startDrag);
        svg.addEventListener('mousemove', drag);
        svg.addEventListener('mouseup', stopDrag);
        svg.addEventListener('mouseleave', stopDrag);
    }
}

let svg;
let scale = 1;
let scaleStep = 0.25;

let isDragging = false;
let startPoint = { x: 0, y: 0 };
let currentTranslation = { x: 0, y: 0 };

function zoom(event) {
    event.preventDefault();
    const direction = event.deltaY < 0 ? 1 : -1;
    scale += direction * scaleStep;
    scale = Math.min(Math.max(scale, 0.1), 5);
    svg.setAttribute('transform', `translate(${currentTranslation.x}, ${currentTranslation.y}) scale(${scale})`);
}

function startDrag(event) {
    isDragging = true;
    startPoint = { x: event.clientX, y: event.clientY };
}

function drag(event) {
    if (!isDragging) return;
    const dx = event.clientX - startPoint.x;
    const dy = event.clientY - startPoint.y;
    currentTranslation = { x: currentTranslation.x + dx / scale, y: currentTranslation.y + dy / scale };
    svg.setAttribute('transform', `translate(${currentTranslation.x}, ${currentTranslation.y}) scale(${scale})`);
    startPoint = { x: event.clientX, y: event.clientY };
}

function stopDrag() {
    isDragging = false;
}
