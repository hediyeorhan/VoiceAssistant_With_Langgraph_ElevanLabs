# Voice Assistant with ElevanLabs and Langgraph

Bu çalışmada Google AI tarafından geliştirilen yapay zekâ Gemini API'ı kullanılarak RAG ve ElevanLabs ile bir sesli asistan projesi geliştirilmiştir. 

Projede __.env__ dosyasında içeriğinde şu veriler bulunmaktadır.

• GEMINI_API_KEY=

• LANGCHAIN_API_KEY=

• LANGCHAIN_TRACING_V2=true

• LANGCHAIN_PROJECT=PROJECT_NAME

• TAVILY_API_KEY=

• ELEVENLABS_API_KEY=

• PDF_PATH = "documents"

• VOICE_ID = ""

• MODEL_NAME = ""

• TOPIC = "hediye orhan kişisi"

• CHUNK_SIZE= 1000

• CHUNK_OVERLAP= 400

• EMBEDDING_MODEL= "models/text-embedding-004"

Projede, Gemini AI ile birlikte Langchain ve Langgraph framework'ü kullanılmıştır. Langchain, büyük dil modelleri ile uygulama geliştirilmesinde kullanılmaktadır. Zincir yapısında LLM'lerin birbirleri ile ve insanlar ile konuşmasını sağlamaktadır. Doküman okuma-yükleme, chat geçmişi tutma, embedding işlemleri ve vektör database işlemleri için langchain framework'ünden faydalanılmıştır. LangChain, LLM'ler ile entegrasyon sağlayarak özelleştirilmiş sorgu yönetimi sunmaktadır. Langgraph ise agent oluşturma, chat hafızasını bellekte / veri tabanında tutma gibi hizmetler sunmaktadır.

<h3> TAVILY </h3>

<br>
Bu çalışmada, Tavily kullanılarak LLM modelinin web sayfası araştırmaları ile entegre bir şekilde bir agent yapısında çalışması sağlanmıştır.


<br>

Langgraph kullanılarak chat hafızası bir veri tabanı dosyasına kayıt edilmiştir. Bu sayede eski chat konuşmaları kaybolmamıştır ve geliştirilen model daha tutarlı sonuçlar / cevaplar üretmiştir.

<br>

Proje bir graph yapısında geliştirilmiştir. Bu graph yapısında öncelikle belirlediğimiz pdf dosyalarından çekilen veriler lokalde oluşturulan bir db'de vectorstore olarak tutulmaktaadır. Oluşturulan sistem öncelikle sorulan soruya ait uygun cevabı vectorstore içerisinde aramaktadır. Vectorstore'da bulunamayan cevaplar Tavily aracılığı ile websearch ile aranmaktadır. Oluşturulan grap yapısında kullanıcının sorduğu soru vectorstore, web veya gemini api-key aracılığıyla yanıt bulmaktadır. Bunlardan hangisi ile kullanıcıya nasıl cevap verileceğini grap yapısı kendisi karar vermektedir.

Çalışmada oluşturulan graph yapısı Şekil 1'de görülmektedir.
<br>
<br>
<div align="center">
<img src="https://github.com/user-attachments/assets/e92cc762-50c1-49c4-a880-865b474ce942" alt="image">
</div>
Şekil 1. Graph yapısı
<br>
<br>

Vectorstore veya websearch'ten elde edilen cevaplar grader fonksiyonlarına gönderilmektedir. Retrieval grader ile dokumandandaki bilgiler ve alınan cevap tutarlı mı kontrol edilmektedir. Kontrol sonucunda string olarak "yes" ve "no" cevapları dönmektedir. Halüsinasyon grader ile ise llm'in halüsinasyon görüp görmediği kontrol edilmektedir. Kontrol sonucunda binary "0" ve "1" olarak bir cevap dönmektedir. Halüsinasyon görülmüyorsa cevabın sorulan soru ile tutarlı olup olmadığı kontrolü yapılarak kullanıcıya bir sonuç dönülmektedir.

Bu aşamalar sonucunda kod Şekil 1'de graph yapısındaki gibi bir yol izlemektedir. Graph yapısındaki düz çizgiler koşulsuz edge, kesikli çizgiler ise koşula bağlı edge olarak tanımlanmaktadır. Burada tanımlanan koşullar grader ve router fonksiyonlarıdır.
<br>

Oluşturulan graph yapısı ile kullanıcıya daha iyi bir kullanım sunmak amacıyla __Gradio__ ile bir arayüz oluşturulmuştur. Bu arayüz üzerindeki mikrofon işareti yardımıyla kullanıcı sorularına cevap bulmaktadır.

<br>
<br>

Kullanıcının oluşturulan asistan ile sesli ve rahat bir şekilde iletişim kurması için __ElevanLabs__ kullanılmıştır. Buradan seçilen türk bir kullanıcının sesi ile kullanıcıya verilen cevaplar seslendirilmiştir. Çalışmada text-to-speech ve speech-to-text fonksiyonları yer almaktadır.

Çalışmanın örnek çıktısı __output_example.mp4__ dosyasında mevcuttur.
📽️ [Videoyu buradan izleyin](https://drive.google.com/file/d/1p0PE2NVZ8IwdWPjONgGrjKieSvRM65Fc/view?usp=sharing)

