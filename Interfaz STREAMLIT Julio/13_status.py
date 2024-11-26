import streamlit as st
import time
from random import choice, randint 

st.title(":mag: Real-Time Status Monitoring Dashboard") 

col1, col2, col3 = st.columns(3) 

#Panekl de notificaciones generales 
with col1:
    st.header("General Notification")
    st.success(":white_check_mark: All systems operational!")
    st.info(":information_source: System update scheduled for tonight.")

with col2:
    st.header("Alerts & Warnings")
    st.warning(":warning: CPU Usage reaching high levels.")
    st.error(":x: Server 3 is not responding.")

with col3:
    st.header("System Exception")
    st.exception(RuntimeError("RuntimeError: Failed to load configuration file.")) 

#Simulación de actualización de datos cada 2 segundos 
st.subheader(":bar_chart: Live Status Updates")
status_area = st.empty()

for _ in range(10):
    update_type = choice(["success","warning","error","info"])
    message = {
        "success": f":white_check_mark: All systems stable at {time.strftime('%H:%M:%S')}",
        "warning": f":warning: Memory usage at {randint(80,95)}%!",
        "error": f":x: Critical error in Service {randint(1,5)}%!",
        "info": f":information_source: Routine maintence scheduled."
    }

with status_area.container():
    getattr(st, update_type)(message[update_type])
time.sleep(2) 

