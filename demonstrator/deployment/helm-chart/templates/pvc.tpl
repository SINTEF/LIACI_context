{{- if not .Values.persistence.existingClaim }}
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: {{ include "liaci-context.fullname" . }}
  labels:
    {{- include "liaci-context.labels" . | nindent 4 }}
spec:
  accessModes:
    - {{ .Values.persistence.accessMode | quote }}
  resources:
    requests:
      storage: {{ .Values.persistence.size | quote }}
  {{- if .Values.persistence.storageClass }}
  storageClassName: {{ .Values.persistence.storageClass }}
  {{- end }}
{{- end }}