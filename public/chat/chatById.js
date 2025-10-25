document.addEventListener("DOMContentLoaded", async () => {
    
     /**
      * 1) Get Information of Chat and User
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
     chatId = window.location.pathname.split('/')[2]
     userId = "TEST_USER"

     // 2)
     

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
    const loadChatFunction = async () => {
      submitButton.addEventListener('click', async (e) => {
        e.preventDefault()

        // Disable Submit Button
        disableButton(submitButton)

        // Extract and Empty the Question box
        const questionInputElement = document.querySelector('#input-question') 
        const question = questionInputElement.value
        questionInputElement.value = ''


        // Create a new Chat Item
        const now = new Date();
        const dateCreated = now.toUTCString();

        const chatSpaceList = document.querySelector('#chat-space')
        const itemElement = document.createElement('section')
            itemElement.classList.add('row');
            itemElement.innerHTML =`
            <section class="container chat-item">
              <section class="row question-sect">
                <section class="container user-sect">
                  <section class="row userinfo-sect">
                    <p><b>${userId}</b> - <span>${dateCreated}</span></p>
                  </section>
                  <section class="row question-sect"><p>${question}</p></section>
                </section>
              </section>
              <section class="row response-sect">
                <section class="container ai-sect">
                  <section class="row aiinfo-sect">
                    <p><b>LLM (Ollama)</b> - <span>${dateCreated}</span></p>
                  </section>
                  <section class="row response-sect">
                    <p class="genereated-result-text">Generating Response...</p>
                  </section>
                </section>
              </section>
            </section>
          `

          chatSpaceList.appendChild(itemElement)

        const formData = JSON.stringify({
            question: question
        })

        const targetEndpoint = `/v1/inference`
        const result = await fetch(targetEndpoint, {
          method: "POST",
          headers: {
            "Content-type": "application/json"
          },
          body: formData
        })

        const data = await result.json()

        if(!result.ok){
          alert(data.message)
          enableButton(submitButton)
          throw new Error(data.message)
        }

        // Extract the response
        const response = data.data.response

        // Display the response
        const generatedResultTextElement = itemElement.querySelector(".genereated-result-text")
        generatedResultTextElement.textContent = response
        
        // Re-enable the Button
        enableButton(submitButton)

      })
    }
    loadChatFunction()
})