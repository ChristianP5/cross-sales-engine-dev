document.addEventListener("DOMContentLoaded", async () => {
    
     /**
      * 1) Check if the Chat ID Exists
      * 2) Generate Inferences from that Chat
      * 3) Load the Chat Function
      */

     // Initial State
     const disableButton = (button) => {
        button.textContent = "Please Wait"
        button.disabled = true
    }

    const enableButton = (button) => {
        button.textContent = "Submit"
        button.disabled = false
    }

    const submitButton = document.querySelector('#chat-submit-btn')
    disableButton(submitButton)

     // 1)

     // 2)
     chatId = window.location.pathname.split('/')[2]

     const generateInferencesByChatId = async (chatId) => {
        const targetEndpoint = `/v1/chats/${chatId}/inferences`
        const result = await fetch(targetEndpoint, {
            method: 'GET'
        })

        const data = await result.json()

        if(!result.ok){
            alert(data.message)
            throw new Error(data.message)
        }

        inferences =  data.data.inferences
        // console.log(inferences)

        const chatSpaceList = document.querySelector('#chat-space')
        inferences.forEach(inference => {
            const itemElement = document.createElement('section')
            itemElement.classList.add('row');
            itemElement.innerHTML =`
            <section class="container chat-item">
              <section class="row question-sect">
                <section class="container user-sect">
                  <section class="row userinfo-sect">
                    <p><b>${inference.userId}</b> - <span>${inference.dateCreated}</span></p>
                  </section>
                  <section class="row question-sect"><p>${inference.initialPrompt}</p></section>
                </section>
              </section>
              <section class="row response-sect">
                <section class="container ai-sect">
                  <section class="row aiinfo-sect">
                    <p><b>LLM (Ollama)</b> - <span>${inference.dateCreated}</span></p>
                  </section>
                  <section class="row response-sect">
                    <p>${inference.response}</p>
                  </section>
                </section>
              </section>
            </section>
          `

          chatSpaceList.appendChild(itemElement)
        });

     }
     await generateInferencesByChatId(chatId)
     enableButton(submitButton)

     // 3)
     const loadChatFunction = async () => [

     ]
})