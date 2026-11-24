from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Elon Reeve Musk (EE-lon; lahir 28 Juni 1971) adalah seorang pengusaha dan mantan pejabat publik yang dikenal karena kepemimpinannya di Tesla dan SpaceX. Musk menjadi orang terkaya di dunia sejak tahun 2025; pada Juni 2026, Forbes memperkirakan kekayaan bersihnya mencapai US$834 miliar. Pada 12 Juni 2026—bertepatan dengan penawaran umum perdana SpaceX—Bloomberg memperkirakan kekayaannya telah mencapai US$1,11 triliun, menjadikannya triliuner pertama di dunia.[3]

Setelah mengikuti kuliah perdagangan di Queen's University selama dua tahun (1989-1991), Musk pindah ke Amerika Serikat. Ia mendapat gelar sarjana fisika dari Universitas Pennsylvania, lalu gelar sarjana ekonomi dari The Wharton School.[4] Ia melanjutkan studi doktoral Fisika di Universitas Stanford pada 1995, tetapi malah keluar dan mendirikan Zip2 bersama saudaraya, Kimbal Musk. Dia juga memiliki kontribusi dalam perusahaan PayPal.[5][6]

Elon Musk berperan dalam mendirikan perusahaan SpaceX yang bergerak di teknologi antariksa; Tesla, Inc. yang bergerak di bidang otomotif; OpenAI, SolarCity, Neuralink, dan beberapa perusahaan lain.

Elon Musk memiliki peran besar dalam pemilihan Donald Trump pada Pemilihan umum Presiden Amerika Serikat 2024. Dia dikenal sebagai pendukung tokoh, gerakan, dan partai politik sayap kanan jauh di berbagai belahan dunia, seperti Presiden Argentina Javier Milei, anggota parlemen Belanda Geert Wilders, dan partai politik Jerman AfD.

Pada awal 2025, ia menjabat sebagai Penasihat Senior Presiden Amerika Serikat Donald Trump dan secara de facto menjadi kepala DOGE. Setelah terjadi perseteruan publik dengan Trump, Musk keluar dari pemerintahan Trump dan mengumumkan pembentukan partai politiknya sendiri, yaitu Partai Amerika.

Masa kecil
Elon lahir di Pretoria, Afrika Selatan, dari pasangan ibu Kanada dan ayah Afrika Selatan.[7][8][9] Ibunya bernama Maye Musk, seorang model dari Regina, Saskatchewan, Kanada, sementara ayahnya adalah Errol Musk, seorang ahli elektromekanika, pilot, dan pelaut.

Ia memiliki seorang adik laki-laki bernama Kimbal (lahir pada 1972) dan adik perempuan, Tosca (lahir 1974). Neneknya berasal dari Britania Raya, sementara kakeknya berasal dari Amerika Serikat. Setelah orang tuanya bercerai pada tahun 1980, Elon tinggal bersama ayahnya di Pretoria.

Sejak kecil, Elon sangat senang membaca dan belajar. Pada usia 10 tahun, ia mulai tertarik dengan mempelajari teknik komputer. Pada usia 12 tahun, ia belajar pemrograman dan berhasil membuat serta menjual kode komputer pada majalah komputer untuk sebuah game yang ia beri nama Blastar[10] dengan harga $500.[11]

Elon lulus dari Pretoria Boys High School dan pindah ke Kanada tahun 1989 pada usia 17 tahun untuk menghindari wajib militer di Afrika Selatan.[11][12] Tahun 1992, setelah belajar selama dua tahun di Queen's University, Kingston, Ontario, Musk pindah ke Wharton School di Pennsylvania dan memperoleh gelar S1 dalam bidang ekonomi dan S1 kedua dalam bidang fisika.[4][5] Ia pindah ke California untuk mengejar gelar Ph.D. dalam bidang fisika terapan di Stanford, tetapi keluar setelah dua hari demi mengejar keinginannya untuk berwiraswasta di sektor internet, energi terbarukan, dan luar angkasa pada usia 24 tahun.[11][13]
    """

    summary_template = """
    given the information: {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables={"information"}, template=summary_template
    )

    # print(summary_prompt_template.format(information=information))

    # llm = ChatOpenAI(temperature=0, model="gpt-4o")

    # llm2 = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    # llm3 = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

    llm4 = ChatOllama(model="gemma3:270m", temperature=0)
    # llm5 = ChatOllama(model="qwen3.8", temperature=0)
    llm6 = ChatOllama(model="gemma3", temperature=0)
    # llm7 = ChatOllama(model="gpt-oss", temperature=0)
    # llm8 = ChatOllama(model="hermes3", temperature=0)
    llm9 = ChatOllama(model="qwen3:4b", temperature=0)

    chain = summary_prompt_template | llm9

    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
