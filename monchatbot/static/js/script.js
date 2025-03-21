function addMessage(message, isUser = false) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();

    if (message) {
        addMessage(message, true);
        input.value = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (!data || !data.response) {
                throw new Error('Réponse invalide du serveur');
            }

            addMessage(data.response);
        } catch (error) {
            //console.error('Error:', error);
            addMessage('Désolé, une erreur est survenue. ' + 
                      'Veuillez réessayer dans quelques instants. ' +
                      'Code erreur: ' + error.message);
        }
    }
}

// Amélioration de la fonction addMessage pour plus de robustesse
function addMessage(message, isUser = false) {
    try {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) {
            console.error('Element chat-messages non trouvé');
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        
        // Échappement du HTML pour la sécurité
        const textNode = document.createTextNode(message);
        messageDiv.appendChild(textNode);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Erreur dans addMessage:', error);
    }
}


document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});