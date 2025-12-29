document.addEventListener('DOMContentLoaded', async () => {
    console.log("Working!")

    // Initialize the Regulation Documents List
    const initializeRegulationsList = async () => {
        /**
        1) Get Documents
        2) Display Documents 
        */

        const loadRegulationsDocs_targetEndpoint = "/v1/regulations"
        const response = await fetch(loadRegulationsDocs_targetEndpoint, {
            method: "GET"
        })

        const data = await response.json()

        if(!response.ok){
            throw new Error(data)
        }

        const docs = data.data.docs

        console.log(docs)

        const regulationsInventoryListElement = document.querySelector('#regulations-inventory-list')
        docs.forEach(doc => {
            const item = document.createElement("tr")
            item.innerHTML = `
            <td>${doc.id}</td>
            <td>${doc.type}</td>
            <td>${doc.name}</td>
            <td>${doc.createdAt}</td>
            <td>
                <button type="button" class="btn btn-danger delete-doc-btn">Delete</button>
                <button type="button" class="btn btn-success download-doc-btn">Download</button>
            </td>
            `

            deleteDocumentButton = item.querySelector(".delete-doc-btn")
            deleteDocumentButton.addEventListener("click", async (e) => {
                e.preventDefault()

                const deleteDocument_targetEndpoint = `/v1/documents/${doc.id}`
                const response = await fetch(deleteDocument_targetEndpoint, {
                    method: "DELETE"
                })

                const data = await response.json()

                if(!response.ok){
                    alert(data.message)
                    throw new Error(data)
                }

                alert(`Document ${doc.name} deleted successfully!`)
                location.href = `/inventory`
                return;

            })

            regulationsInventoryListElement.appendChild(item)
        })
    }
    initializeRegulationsList()

    // Initialize the Customers Documents List
    const initializeCustomerDocsList = async () => {
        /**
        1) Get Documents
        2) Display Documents 
        */

        const loadCustomerDocs_targetEndpoint = "/v1/customerDocs"
        const response = await fetch(loadCustomerDocs_targetEndpoint, {
            method: "GET"
        })

        const data = await response.json()

        if(!response.ok){
            throw new Error(data)
        }

        const docs = data.data.docs

        console.log(docs)

        const customerDocsInventoryListElement = document.querySelector('#customer-inventory-list')
        docs.forEach(doc => {
            const item = document.createElement("tr")
            item.innerHTML = `
            <td>${doc.id}</td>
            <td>${doc.type}</td>
            <td>${doc.name}</td>
            <td>${doc.createdAt}</td>
            <td>
                <button type="button" class="btn btn-danger delete-doc-btn">Delete</button>
                <button type="button" class="btn btn-success download-doc-btn">Download</button>
            </td>
            `

            deleteDocumentButton = item.querySelector(".delete-doc-btn")
            deleteDocumentButton.addEventListener("click", async (e) => {
                e.preventDefault()

                const deleteDocument_targetEndpoint = `/v1/documents/${doc.id}`
                const response = await fetch(deleteDocument_targetEndpoint, {
                    method: "DELETE"
                })

                const data = await response.json()

                if(!response.ok){
                    alert(data.message)
                    throw new Error(data)
                }

                alert(`Document ${doc.name} deleted successfully!`)
                location.href = `/inventory`
                return;

            })

            customerDocsInventoryListElement.appendChild(item)
        })
    }
    initializeCustomerDocsList()

    // Initialize the Products Documents List
    const initializeProductDocsList = async () => {
        /**
        1) Get Documents
        2) Display Documents 
        */

        const loadCustomerDocs_targetEndpoint = "/v1/productDocs"
        const response = await fetch(loadCustomerDocs_targetEndpoint, {
            method: "GET"
        })

        const data = await response.json()

        if(!response.ok){
            throw new Error(data)
        }

        const docs = data.data.docs

        console.log(docs)

        const customerDocsInventoryListElement = document.querySelector('#product-inventory-list')
        docs.forEach(doc => {
            const item = document.createElement("tr")
            item.innerHTML = `
            <td>${doc.id}</td>
            <td>${doc.type}</td>
            <td>${doc.name}</td>
            <td>${doc.createdAt}</td>
            <td>
                <button type="button" class="btn btn-danger delete-doc-btn">Delete</button>
                <button type="button" class="btn btn-success download-doc-btn">Download</button>
            </td>
            `

            deleteDocumentButton = item.querySelector(".delete-doc-btn")
            deleteDocumentButton.addEventListener("click", async (e) => {
                e.preventDefault()

                const deleteDocument_targetEndpoint = `/v1/documents/${doc.id}`
                const response = await fetch(deleteDocument_targetEndpoint, {
                    method: "DELETE"
                })

                const data = await response.json()

                if(!response.ok){
                    alert(data.message)
                    throw new Error(data)
                }

                alert(`Document ${doc.name} deleted successfully!`)
                location.href = `/inventory`
                return;

            })

            customerDocsInventoryListElement.appendChild(item)
        })
    }
    initializeProductDocsList()



    // Initialize the Customer Input
    var customers
    const initializeCustomerInput = async () => {
        const customerInputElement = document.querySelector("#customer-input")

        const targetEndpoint = "/v1/customers"
        const result = await fetch(targetEndpoint, {
            method: "GET"
        })

        const data = await result.json()

        if(!result.ok){
            alert(data.message)
            throw new Error(data.message)
        }

        customers = data.data.customers

        customers.forEach(customer => {
            const item = document.createElement("option")
            item.value = customer.customerId
            item.textContent = customer.name

            customerInputElement.appendChild(item)
            customerInputElement.value = item.value
        })

    }
    await initializeCustomerInput()

    // Configure Upload 
    
    const disableButton = (button) => {
        button.textContent = "Please Wait"
        button.disabled = true
    }

    const enableButton = (button) => {
        button.textContent = "Submit"
        button.disabled = false
    }

    // for assigning the 'purpose' Value to the Upload Request
    const purposeValueElement = document.querySelector("#purpose-actual-input")

    const initializeUpload = async () => {
        const uploadForm = document.querySelector("#upload-form")
        const uploadSubmitButton = uploadForm.querySelector("#submit-btn")

        

        // Display the Customer Input Element if Purpose == CUSTOMER
        const purposeInputElement = document.querySelector("#purpose-input")
        var currentPurposeValue = purposeInputElement.value

        purposeInputElement.addEventListener("change", async (e) => {

            currentPurposeValue = purposeInputElement.value
            console.log(currentPurposeValue)
            
            const customerInputSection = document.querySelector("#customer-input-section")

            if(currentPurposeValue != "REGULATION" && currentPurposeValue != "PRODUCT"){
                customerInputSection.classList.remove("d-none")
                customerInputElement.value = customers[0].customerId
                currentPurposeValue = customers[0].customerId
            }else{
                customerInputSection.classList.add("d-none")
            }
        })

        // Set customerId value when a Customer is selected
        const customerInputElement = document.querySelector("#customer-input")

        customerInputElement.addEventListener("change", async (e) => {

            currentPurposeValue = customerInputElement.value
            console.log(currentPurposeValue)
            
        })

        uploadSubmitButton.addEventListener("click", async (e) => {
            e.preventDefault()

            // Set purpose = customerId
            purposeValueElement.value = currentPurposeValue
            console.log(purposeValueElement)

            disableButton(uploadSubmitButton)

            const formData = new FormData(uploadForm)
            
            /*
            for (const [key, value] of formData.entries()) {
                console.log(key, value);
            }
            */

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