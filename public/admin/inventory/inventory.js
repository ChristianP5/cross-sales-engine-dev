document.addEventListener('DOMContentLoaded', async () => {
    console.log("Working!")

    // Initialize the List
    const initializeList = async () => {
        /**
        1) Get Documents
        2) Display Documents 
        */

        const targetEndpoint = "/v1/documents"
        const response = await fetch(targetEndpoint, {
            method: "GET"
        })

        const data = await response.json()

        if(!response.ok){
            throw new Error(data)
        }

        const docs = data.data.docs

        console.log(docs)

        const inventoryListElement = document.querySelector('#inventory-list')
        docs.forEach(doc => {
            const item = document.createElement("tr")
            item.innerHTML = `
            <td>${doc.id}</td>
            <td>${doc.type}</td>
            <td>${doc.name}</td>
            <td>${doc.date}</td>
            `

            inventoryListElement.appendChild(item)
        })
    }

    initializeList()

    // Configure Upload 
    
    const disableButton = (button) => {
        button.textContent = "Please Wait"
        button.disabled = true
    }

    const enableButton = (button) => {
        button.textContent = "Submit"
        button.disabled = false
    }


    const initializeUpload = async () => {
        const uploadForm = document.querySelector("#upload-form")
        const uploadSubmitButton = uploadForm.querySelector("#submit-btn")
        uploadSubmitButton.addEventListener("click", async (e) => {
            e.preventDefault()

            disableButton(uploadSubmitButton)

            const formData = new FormData(uploadForm)

            const targetEndpoint = "/v1/upload"
            const response = await fetch(targetEndpoint, {
                method: "POST",
                body: formData
            })

            const data = await response.json()

            if(!response.ok){
                alert(data.message)
                enableButton(uploadSubmitButton)
                throw new Error(data.message)
                
            }

            alert(data.message)

            location.href = `/inventory`
            return;
            
        })
    }
    
    initializeUpload()


})