# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #                     APPLICATION PARAMETERS                      # # # # # # # # # # # # # #

# Tamanho máxima da fila de mensagens entre processos
max_queue_size = 5

# Tamanhos mínimos de máximos de uma Face:
min_size = 85
max_size = 400

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #                        MARGIN PARAMETERS                        # # # # # # # # # # # # # #
# Ignora as margens da esquerda e direita - EM PORCENTAGEM DO FRAME (entre 0.0 e 1.0)
ignore_left_margin = 0.15
ignore_right_margin = 0.15

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #                    FACE TRACKING PARAMETERS                     # # # # # # # # # # # # # #
# quantidade de faces/frames armazenadas em CADA track (tamanho do vetor / histórico de faces)
max_track_size = 6
# Quantidade de frames necessários com rosto antes de tentar o reconhecimento
min_frames_before_recognition = 4
# Tempo de vida da track (segundos)
track_life_in_seconds = 3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #                   FACE RECOGNITION PARAMETERS                   # # # # # # # # # # # # # #
# Tempo de vida das tracks processadas/ reconhecidas
track_recogn_life_in_seconds = 3


# Tentativas de reconhecimento offline
max_offline_recognition_attmeps = 3
# Probabilidade  exegida no reconhecimento offline
min_confidence_level_offiline = 99.75

# Número de threads dedicadas às requisições online
number_threads_aws = 3
# Time out da requisição (em segundos)
time_out = 3.5
# Quantidade máxima de requisições enviadas à AWS
max_recognition_attempts = 1
# Intervalo de frames entre as tentativas de reconhecimento online
frames_to_skip_before_recognize = 3
# Porcentagem de incremento do bounding box ( ** EM PORCENTAGEM)
increase_face_bbox = 0.4

# BLoqueia a track quando não encontra faces
block_duration_in_seconds = 10.0

# Antifraude parameters
# Número máximo de processos iniciados para processamento antifraude
number_processes_antifraude = 3
# Processa até X frames para confirmar se a face é FAKE
max_antifraude_attemps = 12
# Janela de busca das faces
antifraude_window_length = 6
# Número de resultados REAIS para confirmar face REAL
number_to_confirm_real_face = 2

# Número máximo de faces salvos no banco por ID
max_number_faces = 15
