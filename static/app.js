const sendBtn = document.getElementById("send-btn");
const promptInput = document.getElementById("prompt");
const messages = document.getElementById("messages");

function addMessage(text, role) {
    const div = document.createElement("div");
    div.classList.add("message", role);
    div.innerText = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

async function sendMessage() {

    const text = promptInput.value.trim();
    if (!text) return;

    addMessage(text, "user");
    promptInput.value = "";

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: text})
        });

        const data = await res.json();
        addMessage(data.reply, "assistant");

    } catch (err) {
        addMessage("Error connecting to server", "assistant");
    }
}

sendBtn.onclick = sendMessage;

promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});