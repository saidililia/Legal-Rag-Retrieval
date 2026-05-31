const sendBtn = document.getElementById("send-btn");
const promptInput = document.getElementById("prompt");
const messages = document.getElementById("messages");

function addMessage(text, sender) {

    const div = document.createElement("div");

    div.classList.add("message");
    div.classList.add(sender);

    div.innerText = text;

    messages.appendChild(div);

    messages.scrollTop = messages.scrollHeight;

    return div;
}


async function sendMessage() {

    const text = promptInput.value.trim();

    if (!text) return;

    // Add user message
    addMessage(text, "user");

    promptInput.value = "";

    // Create empty assistant message
    const assistantDiv = addMessage("", "assistant");

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            message: text
        })
    });

    const reader = response.body.getReader();

    const decoder = new TextDecoder();

    while (true) {

        const { done, value } = await reader.read();

        if (done) break;

        const chunk = decoder.decode(value);

        assistantDiv.innerText += chunk;

        messages.scrollTop = messages.scrollHeight;
    }
}


sendBtn.addEventListener("click", sendMessage);

promptInput.addEventListener("keydown", (e) => {

    if (e.key === "Enter" && !e.shiftKey) {

        e.preventDefault();

        sendMessage();
    }
});