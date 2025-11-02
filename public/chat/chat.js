document.addEventListener('DOMContentLoaded', async () => {
    /**
     * 1) Implement Toggle Inference Version feature
     * 2) Implement Inference feature
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

        })
    }

    await inferencefeature()
})