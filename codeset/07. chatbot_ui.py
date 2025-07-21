import streamlit as st
import requests

st.set_page_config(page_title="FaMiliCare 건강 챗봇", page_icon="💬")

st.title("FaMiliCare 건강 챗봇")
st.write("member_id를 입력하고 진단 결과를 받아보세요.")

member_id = st.text_input("member_id", value="52a62c11-7c4f-4912-91c9-ff5c145328cd")
llm = st.selectbox("LLM 모델 선택", options=["gpt", "ollama"], index=0)
if st.button("진단 결과 요청"):
    if not member_id:
        st.warning("member_id를 입력하세요.")
    else:
        with st.spinner("진단 결과를 불러오는 중..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:9000/generate_report",
                    json={"member_id": member_id, "llm": llm},
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json().get("result", "")
                    st.success("진단 결과:")
                    st.markdown(result)
                else:
                    st.error(f"오류: {response.json().get('detail', response.text)}")
            except Exception as e:
                st.error(f"서버 연결 오류: {e}") 