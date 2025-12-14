document.addEventListener("DOMContentLoaded", async () => {
    
     /**
      * 1) Get Information of Chat and User
      * 2) Generate Chats fom that User
      * 2) Generate Inferences from that Chat
      * 3) Load the Chat Function
      * 5) Load the Create Chat Function
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

    const scrollMax = (container) => {
      container.scrollTop = container.scrollHeight;
    }

    const submitButton = document.querySelector('#chat-submit-btn')
    disableButton(submitButton)

     // 1)
     chatId = window.location.pathname.split('/')[2]
     userId = "TEST_USER"

     // 2)
     const generateChatsByUserId = async(userId) => {
      const targetEndpoint = '/v1/chats'
      const result = await fetch(targetEndpoint, {
        headers: {
          'Authorization': `Bearer ${userId}`
        },
        method: 'GET'
      })

      const data = await result.json()

      if(!result.ok){
        alert(data.message)
        throw new Error(data.message)
      }

      const chatList = document.querySelector("#chat-list")
      const chats = data.data.chats

      chats.forEach(chat => {
        const item = document.createElement("div")
        item.classList.add("row")
        item.innerHTML = `
        <span id="chat-item">${chat.name}</span>
        `
        chatList.appendChild(item)

        const clickable = item.querySelector("#chat-item")
        clickable.addEventListener("click", async (e) => {
          e.preventDefault()

          window.location.href = `/chats/${chat.chatId}`
          return;
        })

      })
      



     }
     await generateChatsByUserId(userId)

     // 3)
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

        const chatSectionScrollable = document.querySelector("#chat-sect")
        const chatSpaceList = document.querySelector('#chat-space')
        inferences.forEach(inference => {
            const itemElement = document.createElement('section')
            itemElement.classList.add('row');
            itemElement.classList.add('bg-white');
            itemElement.classList.add('m-2');
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
                  <section class="row" id="chat-response-docs-sect">
                      <div class="col-1 align-content-center">
                        <p>Sources:</p>
                      </div>

                      <div
                        id="chat-response-docs-list"
                        class="col-11 d-flex flex-wrap gap-2 mt-2 align-items-center"
                      >
                      </div>
                    </section>
                    <p>${inference.response_html}</p>
                  </section>
                </section>
              </section>
            </section>
          `
          const chatResponseDocsList = itemElement.querySelector("#chat-response-docs-list")
          const retrieved_docs = inference.docs

          retrieved_docs.forEach(doc => {
            const item = document.createElement("div")
                item.classList.add("source-box", "d-flex", "align-items-center", "p-2")
                
                var image_src = "new-document.png"

                if(doc.type == "PDF"){
                    image_src = "pdf.png"
                }

                item.innerHTML = `
                    <img
                    class="source-icon"
                    src="/file/_assets/${image_src}"
                    alt=""
                    />
                    <span class="ms-2">${doc.name}</span>
                `

                chatResponseDocsList.appendChild(item)

          })

          chatSpaceList.appendChild(itemElement)

          // Scroll the Chat Section down
          scrollMax(chatSectionScrollable)
          
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
            itemElement.classList.add('bg-white');
            itemElement.classList.add('m-2');
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
          scrollMax(chatSectionScrollable)

        const formData = JSON.stringify({
            question: question,
            chatId: chatId
        })

        const targetEndpoint = `/v1/chat`
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
        const response = data.data.response_html

        // Display the response
        const generatedResultTextElement = itemElement.querySelector(".genereated-result-text")
        generatedResultTextElement.innerHTML = response
        
        // Scroll the Chat Section down
        const chatSectionScrollable = document.querySelector("#chat-sect")
        scrollMax(chatSectionScrollable)


        // Re-enable the Button
        enableButton(submitButton)

      })
    }
    loadChatFunction()

    // 5)
    const loadCreateChatFunction = async () => {
      const createChatButton = document.querySelector("#create-chat-btn")
      createChatButton.addEventListener("click", async (e) => {
        e.preventDefault()

        const createChatNameInput = document.querySelector("#create-chat-name-input")
        const chatName = createChatNameInput.value

        const formData = JSON.stringify({
          name: chatName
        })

        const targetEndpoint = '/v1/chats/create'
        const result = await fetch(targetEndpoint, {
          headers: {
            "Authorization": `Bearer ${userId}`,
            "Content-type": "application/json"
          },
          method: 'POST',
          body: formData

        })

        const data = await result.json()

        if(!result.ok){
          alert(data.message)
          throw new Error(data.message)
        }

        alert(data.message)
        
        const new_chatId = data.data.chatId
        window.location.href = `/chats/${new_chatId}`
        return 0

      })
    }
    await loadCreateChatFunction()
})