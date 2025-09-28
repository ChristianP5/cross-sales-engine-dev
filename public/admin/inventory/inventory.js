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
            throw new Error(response)
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
            <td>${doc.date}}</td>
            `

            inventoryListElement.appendChild(item)
        })
    }

    initializeList()

    // Configure Upload Behavior
    
    

})