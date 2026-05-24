const sendBtn = document.getElementById("send-btn");
const promptInput = document.getElementById("prompt");
const messages = document.getElementById("messages");


function addMessage(text, role) {

    const div = document.createElement("div");

    div.classList.add("message");
    div.classList.add(role);

    div.innerText = text;

    messages.appendChild(div);

    messages.scrollTop = messages.scrollHeight;
}


async function sendMessage() {

    const text = promptInput.value.trim();

    if (!text) return;

    // Add user message
    addMessage(text, "user");

    // Clear input
    promptInput.value = "";

    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: text
            })
        });

        const data = await response.json();

        // Add AI response
        addMessage(data.reply, "assistant");

    } catch (error) {

        addMessage("Error connecting to server.", "assistant");

        console.error(error);
    }
}


// Button click
sendBtn.addEventListener("click", sendMessage);


// Enter key
promptInput.addEventListener("keydown", (e) => {

    if (e.key === "Enter" && !e.shiftKey) {

        e.preventDefault();

        sendMessage();
    }
});