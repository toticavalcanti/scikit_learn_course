from sklearn.metrics import classification_report, confusion_matrix

valores_reais    = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
valores_preditos = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

target_names = ['Não Grávidas', 'Grávidas']
print(classification_report(valores_reais, valores_preditos, target_names=target_names))
print(confusion_matrix(valores_reais, valores_preditos))