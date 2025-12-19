document.addEventListener("DOMContentLoaded", async () => {

    // Extract Customer ID
    const customerId = window.location.pathname.split('/')[2]
    
    /**
     * 1) Load the Customer's Data
     * 2) Load the Documents for the Customemr
     * 
     */

    // 1)
    const loadCustomerData = async () => {
        const loadCustomerData_targetEndpoint = `/v1/customers/${customerId}`
        const result = await fetch(loadCustomerData_targetEndpoint, {
            method: 'GET',
            headers: {
                "Content-type": "application/json"
            }
        })

        const data = await result.json()

        if(!result.ok){
            alert(data.message)
            throw new Error(data)
        }

        const customer = data.data.customer

        // Load the Data to the Page
        const customerNameText = document.querySelector("#customer-name-text")
        const customerProfileText = document.querySelector("#customer-profile-text")
        const customerProductsText = document.querySelector("#customer-products-text")
        const customerContactsText = document.querySelector("#customer-contacts-text")

        customerNameText.textContent = customer.name
        customerProfileText.innerHTML = customer.profile
        customerProductsText.innerHTML = customer.products
        customerContactsText.innerHTML = customer.contacts

    }
    loadCustomerData()

    // 2)
    const loadCustomerDocuments = async () => {
        const loadCustomerDocuments_targetEndpoint = `/v1/customers/${customerId}/documents`
        const result = await fetch(loadCustomerDocuments_targetEndpoint, {
            method: 'GET',
        })

        const data = await result.json()

        if(!result.ok){
            alert(data.message)
            throw new Error(data)
        }

        const docs = data.data.docs

        // Display Document Data to the Page
        const customerDocumentListElement = document.querySelector("#customer-documents-list")

        docs.forEach(doc => {
            const item = document.createElement("tr")
            item.innerHTML = `
            <tr>
                <td>${doc.id}</td>
                <td>${doc.name}</td>
                <td>${doc.createdAt}</td>
                <td>${doc.updatedAt}</td>
                <td>
                    <button type="button" class="btn btn-danger delete-doc-btn">Delete</button>
                    <button type="button" class="btn btn-success download-doc-btn">Download</button>
                </td>
            </tr>
            `

            customerDocumentListElement.appendChild(item)

        })

    }
    loadCustomerDocuments()

})