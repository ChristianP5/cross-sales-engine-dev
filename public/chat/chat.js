document.addEventListener('DOMContentLoaded', async () => {
    /**
     * 1) Implement Toggle Inference Version feature
     * 2) Implement Inference feature
     * 2.a) List Retrieval feature
     * 
     */

    // 1)
    var inferenceVersion = 1
    const inferenceVersion1Button = document.querySelector("#btn-v1")
    const inferenceVersion2Button = document.querySelector("#btn-v2")

    const updateVersionButtons = async (inferenceVersion) => {
        if(inferenceVersion == 1){
            inferenceVersion1Button.classList.replace("btn-secondary", "btn-primary")
            inferenceVersion2Button.classList.replace("btn-primary", "btn-secondary")
        }
        
        if(inferenceVersion == 2){
           inferenceVersion1Button.classList.replace("btn-primary", "btn-secondary")
           inferenceVersion2Button.classList.replace("btn-secondary", "btn-primary") 
        }
    }
    updateVersionButtons(inferenceVersion)

    inferenceVersion1Button.addEventListener("click", async (e) => {
        e.preventDefault()

        inferenceVersion = 1;
        updateVersionButtons(inferenceVersion)
    })

    inferenceVersion2Button.addEventListener("click", async (e) => {
        e.preventDefault()

        inferenceVersion = 2;
        updateVersionButtons(inferenceVersion)
    })

    // 2)
    const disableButton = (button) => {
        button.textContent = "Please Wait"
        button.disabled = true
    }

    const enableButton = (button) => {
        button.textContent = "Submit"
        button.disabled = false
    }

    // 2.a) Start
    const chatResponseDocsSection = document.querySelector("#chat-response-docs-sect")
    chatResponseDocsSection.style.display = "none"
    // 2.a) End

    const inferencefeature = async () => {
        const chatSubmitBtn = document.querySelector('#chat-submit-btn')
        chatSubmitBtn.addEventListener('click', async (e) => {
            e.preventDefault()

            const responseSection = document.querySelector('#chat-response-sect')
            responseSection.textContent = "Please wait..."
            disableButton(chatSubmitBtn)

            const question = document.querySelector('#input-question').value


            const formData = JSON.stringify({
                question: question
            })

            console.log(formData)

            const targetEndpoint = `/v${inferenceVersion}/inference`
            const result = await fetch(targetEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: formData
            })

            const data = await result.json()

            if(!result.ok){
                alert(data.message)
                enableButton(chatSubmitBtn)
                throw new Error(data.message)
            }

            const chatResponse = data.data.response_html

            responseSection.innerHTML = chatResponse

            enableButton(chatSubmitBtn)

            // 2.a)
            chatResponseDocsSection.style.display = ""
            const chatResponseDocsList = document.querySelector("#chat-response-docs-list")

            const retrieved_docs = data.data.docs

            retrieved_docs.forEach( doc => {
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
                    <span class="ms-2">${doc.name}</span>d
                `

                chatResponseDocsList.appendChild(item)

            } )



        })
    }

    await inferencefeature()
})