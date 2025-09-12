# # explain.py
# import shap
# import lime.lime_text
# import numpy as np

# def shap_explain(model, X, vectorizer, top_k=5):
#     explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
#     shap_values = explainer.shap_values(X)

#     top_features = []
#     for i in range(X.shape[0]):
#         row_shap = shap_values[0][i] if isinstance(shap_values, list) else shap_values[i]
#         indices = np.argsort(row_shap)[-top_k:][::-1]
#         words = [vectorizer.get_feature_names_out()[idx] for idx in indices]
#         top_features.append(", ".join(words))
#     return top_features

# def lime_explain(model, vectorizer, text_samples, num_features=5):
#     class_names = model.classes_ if hasattr(model, "classes_") else None
#     explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

#     explanations = []
#     for text in text_samples:
#         exp = explainer.explain_instance(
#             text, 
#             lambda x: model.predict_proba(vectorizer.transform(x)),
#             num_features=num_features
#         )
#         explanations.append(exp.as_list())
#     return explanations
