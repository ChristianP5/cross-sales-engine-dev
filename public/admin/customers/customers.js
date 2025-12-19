document.addEventListener('DOMContentLoaded', async () => {
    console.log("Working!")

    // Initialize the List
    const initializeList = async () => {
        /**
        1) Get Customers
        2) Display Customers 
        */

        const targetEndpoint = "/v1/customers"
        const response = await fetch(targetEndpoint, {
            method: "GET"
        })

        const data = await response.json()

        if(!response.ok){
            throw new Error(data)
        }

        const customers = data.data.customers
        
        console.log(customers)

        const inventoryListElement = document.querySelector('#inventory-list')
        customers.forEach(customer => {
            const item = document.createElement("tr")
            item.innerHTML = `
            <td>${customer.customerId}</td>
            <td>${customer.name}</td>
            <td>${customer.createdAt}</td>
            <td>${customer.updatedAt}</td>
            <td>
                <button type="button" class="btn btn-danger delete-doc-btn">Delete</button>
                <button type="button" class="btn btn-success download-doc-btn">Download</button>
            </td>
            `

            item.addEventListener('click', async (e) => {
                e.preventDefault()

                window.location.href = `/customers/${customer.customerId}`
                return
            })

            inventoryListElement.appendChild(item)
        })
    }

    initializeList()

    // Configure Create Customer 
    
    const disableButton = (button) => {
        button.textContent = "Please Wait"
        button.disabled = true
    }

    const enableButton = (button) => {
        button.textContent = "Create Customer"
        button.disabled = false
    }


    const initializeCreateCustomer = async () => {
        const mainForm = document.querySelector("#main-form")
        const createCustomerButton = mainForm.querySelector("#submit-btn")
        createCustomerButton.addEventListener("click", async (e) => {
            e.preventDefault()

            disableButton(createCustomerButton)

            const nameInputElement = mainForm.querySelector("#input-name")
            const name = nameInputElement.value

            const payload = JSON.stringify({
                name: name
            })

            const targetEndpoint = "/v1/customers/create"
            const response = await fetch(targetEndpoint, {
                method: "POST",
                headers: {
                    "Content-type": "application/json"
                },
                body: payload
            })

            const data = await response.json()

            if(!response.ok){
                alert(data.message)
                enableButton(createCustomerButton)
                throw new Error(data.message)
                
            }

            alert(data.message)

            location.href = `/customers`
            return;
            
        })
    }
    
    initializeCreateCustomer()


})