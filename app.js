const tabs=document.querySelectorAll(".general-tabs");
const dashboardContainer=document.querySelector(".dashboard-content");
const reportContainer=document.querySelector(".report-container");

tabs.forEach((tab, index)=>{
    tab.addEventListener('click',()=>{
        tabs.forEach(tab=>{tab.classList.remove('active')});
        tab.classList.add('active');

        if(index==0){
            reportContainer.classList.remove('active');
            dashboardContainer.classList.add('active');
        }
        else if(index==4){
            dashboardContainer.classList.remove('active');
            reportContainer.classList.add('active');
        }
    })
})