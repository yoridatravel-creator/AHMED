import os
from fastapi import FastAPI, Query
from serpapi import GoogleSearch
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

app = FastAPI()

# 1. إعداد مفاتيح التشغيل
# تأكد من إضافة متاح OpenAI كمتغير بيئة (Environment Variable) في موقع Render باسم OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "ضع_مفتاح_openai_هنا_إذا_لم_تضعه_في_render")

# مفتاح SerpAPI الخاص بك
SERPAPI_KEY = "610cd9a150271202b098302db8e39d24a7e974cebc9c80d0cf8b39d005a9447c"

# 2. بناء أداة البحث المعتمدة على جوجل للطيران (Google Flights)
@tool
def search_google_flights(departure_id: str, arrival_id: str, outbound_date: str) -> str:
    """
    تبحث عن أسعار الطيران الحية ومقارنتها عبر Google Flights.
    المعاملات المطلوبة:
    - departure_id: رمز مطار المغادرة (مثل RUH).
    - arrival_id: رمز مطار الوصول (مثل CAI).
    - outbound_date: تاريخ السفر بصيغة YYYY-MM-DD (مثال: 2026-08-20).
    """
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "currency": "SAR",  # عرض النتائج بالريال السعودي
        "hl": "ar",         # لغة النتائج عربية
        "api_key": SERPAPI_KEY
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # التحقق من وجود رحلات
        best_flights = results.get("best_flights", [])
        if not best_flights:
            best_flights = results.get("other_flights", [])
            
        if not best_flights:
            return "لم يتم العثور على رحلات متاحة لهذا التاريخ والوجهة عبر جوجل للطيران."

        flight_options = []
        # جلب أول 3 خيارات أرخص رحلات
        for index, flight in enumerate(best_flights[:3]):
            price = flight.get("price", "غير معلن")
            
            # حساب تفاصيل الرحلة (الخطوط والمدد)
            flights_detail = flight.get("flights", [{}])[0]
            airline = flights_detail.get("airline", "شركة طيران")
            duration = flight.get("total_duration", 0)  # بالدقائق
            duration_hours = duration // 60
            
            # استخراج الأرقام فقط من السعر لتحويله إلى رقم صحيح وحساب رسوم المكتب
            base_price_str = ''.join(filter(str.isdigit, str(price)))
            base_price = int(base_price_str) if base_price_str else 0
            
            # إضافة رسوم الخدمة الثابتة للمكتب (60 ريال) تلقائياً للسعر المستخرج
            final_price = base_price + 60
            
            flight_options.append(
                f"الخيار {index+1}:\n"
                f"- شركة الطيران: {airline}\n"
                f"- السعر الإجمالي (شامل الرسوم والضريبة): {final_price} ريال سعودي\n"
                f"- مدة الرحلة: {duration_hours} ساعة تقريباً\n"
            )
            
        return "\n".join(flight_options)

    except Exception as e:
        return f"حدث خطأ أثناء الاتصال بجوجل للطيران: {str(e)}"

# 3. إعداد العميل والـ Prompt للذكاء الاصطناعي
tools = [search_google_flights]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """أنت خبير ومساعد سفر ذكي تخدم قطاع تذاكر الطيران وتجلب للمستخدمين أقل الأسعار المتاحة.
    عند استقبال أي طلب، قم بما يلي:
    1. حوّل المدن تلقائياً إلى أكواد المطارات العالمية IATA (مثل: الرياض -> RUH، القاهرة -> CAI).
    2. حوّل التاريخ إلى الصيغة المعيارية YYYY-MM-DD تماشياً مع العام الحالي 2026.
    3. استدعِ الأداة لجلب البيانات الفورية.
    4. نسّق الإجابة النهائية للعميل باللغة العربية بأسلوب لبق ومحترف ومنظم وبأرقام واضحة جداً."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# تجميع وتشغيل العميل
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. نقطة الاتصال (API Endpoint) التي سيناديها شات بيلدر
@app.get("/search-flights")
def api_search_flights(user_message: str = Query(..., description="رسالة العميل القادمة من الواتساب")):
    try:
        result = agent_executor.invoke({"input": user_message})
        return {"response": result["output"]}
    except Exception as e:
        return {"response": f"نعتذر منك، حدث خطأ أثناء معالجة الطلب: {str(e)}"}

# لتشغيل السيرفر محلياً للتجربة (اختياري)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)