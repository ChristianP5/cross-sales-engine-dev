document.addEventListener('DOMContentLoaded', async () => {
    /**
     * 1) Implement Inference feature
     * 
     */

    // 1)
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

            const targetEndpoint = '/v1/inference'
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

            responseSection.textContent = chatResponse

            enableButton(chatSubmitBtn)

        })
    }

    await inferencefeature()
})