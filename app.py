from clinicalner import load_model, W2vWordEmbeddings, annotate_text_as_dict
from pathlib import Path
import gradio as gr

W2vWordEmbeddings = type(
    "W2vWordEmbeddings",
    W2vWordEmbeddings.__bases__,
    dict(W2vWordEmbeddings.__dict__),
)

models = [load_model(model_path) for model_path in Path("models").glob("*.pt")]


def clean_entity_name(entity_name):
    return entity_name.replace("_", " ").capitalize()


def dict_to_gradio(d):
    result = []
    for entity in d["entities"]:
        result.append(
            {
                "entity": clean_entity_name(entity[1]),
                "start": entity[2][0][0],
                "end": entity[2][0][1],
            }
        )
    return result


def annotate_text_as_gradio_dict(text, models):
    result = []
    for model in models:
        result.extend(dict_to_gradio(annotate_text_as_dict(text, model)))
    return result


examples = [
    "El paciente tiene cáncer de pulmón.",
    "amaurosis total y retinosis  pigmentaria con hernia  incisional   umbatico   pelviana +-   4 cm reductible   no es candidata   para cx ambulatoria x que no tiene quien la cuide .",
    "DM INSULINO   REQUIRENTE   HTA   IRC   EN TRATAMIENTO MEDICO   2 BAYPASS  CORONARIO  COLELITIASIS TACO",
    "ACCIDENTE VASCULAR ENCEFALICO AGUDO, NO ESPECIFICADO COMO HEMORRAGICO O ISQUEMICO PACIENTE SECUELADO DE ACV MÚLTIPLES, DEMENCIA, POSTRADO CON TRASTORNO DE LA DEGLUCIÓN MODERADO EN ABRIL  ACTUALMENTE CUADRO DE DISFAGIA SEVERO EN OBSERVACIÓN CON CUADRO",
    "sospecha dhc screenig varices ademaas dolor usuaria de prednisona evaluar ulceras",
]


def ner(text):
    output = annotate_text_as_gradio_dict(text, models)
    return {"text": text, "entities": output}


with gr.Blocks(title="Extractor de entidades clínicas") as demo:
    input = gr.Textbox(placeholder="Ingresa el texto aquí...", label="Texto")
    output = gr.HighlightedText(label="Entidades extraídas")
    gr.Interface(
        ner,
        inputs=input,
        outputs=output,
        title="Extractor de entidades clínicas",
        description="Este modelo extrae entidades clínicas de un texto utilizando modelos de inteligencia artificial entrenados sobre millones de textos clínicos no estructurados.",
        live=True,
        allow_flagging="never",
    )
    gr.Examples(examples=examples, inputs=input, label="Ejemplos")

demo.launch(server_name="0.0.0.0")
