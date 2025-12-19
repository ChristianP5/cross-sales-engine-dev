document.addEventListener("DOMContentLoaded", async () => {
    alert("Working!")

    // Extract Customer ID
    const customerId = window.location.pathname.split('/')[2]
    console.log(customerId)
    
})