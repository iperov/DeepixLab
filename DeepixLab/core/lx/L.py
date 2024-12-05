import re

allowed_langs = {'en' : 'English',
                 'ru' : 'Русский',
                 }

_id_to_string_dict = {

    ##########
    ### COMMON
    ##########
    
    'About':{
        'en' : 'About',
        'ru' : 'О программе',
    },

    'Acceptable_files': {
        'en': 'Acceptable files',
        'ru': 'Принимаемые файлы'
    },
    
    'Action': {
        'en': 'Action',
        'ru': 'Действие'
    },
    'Active': {
        'en': 'Active',
        'ru': 'Активные'
    },
    'Add': {
        'en': 'Add',
        'ru': 'Добавить'
    },
    'Add_delete_points':{
        'en' : 'Add/delete points',
        'ru' : 'Добавить/удалить точки',
    },
    'Add_item': {
        'en': 'Add item',
        'ru': 'Добавить позицию'
    },
    'Apply': {
        'en': 'Apply',
        'ru': 'Применить'
    },
    'Apply_and_reload': {
        'en': 'Apply and reload',
        'ru': 'Применить и перезагрузить'
    },
    'Apply_config' : {
        'en' : 'Apply config',
        'ru' : 'Применить конфигурацию',
    },
    'Application':{
        'en' : 'Application',
        'ru' : 'Приложение',
    },
    'Are_you_sure': {
        'en': 'Are you sure?',
        'ru': 'Вы уверены?'
    },
    'AsyncX_monitor':{
        'en' : 'AsyncX monitor',
        'ru' : 'AsyncX монитор',
    },
    'Augment_pyramid': {
        'en': 'Augment pyramid',
        'ru': 'Пирамидная аугментация'
    },
    'Auto': {
        'en': 'Auto',
        'ru': 'Авто'
    },
    'Average_for':{
        'en' : 'Average for',
        'ru' : 'Среднее для',
    },
    'Backup': {
        'en': 'Backup',
        'ru': 'Резервная копия'
    },
    'Base_dimension' : {
        'en' : 'Base dimension',
        'ru' : 'Базовая размерность',
    },
    'Batch_size' : {
        'en' : 'Batch size',
        'ru' : 'Размер батча',
    },
    'Batch_acc' : {
        'en' : 'Batch accumulation',
        'ru' : 'Аккумуляция батча',
    },
    'Begin': {
        'en': 'Begin',
        'ru': 'Начало'
    },
    'Blue': {
        'en': 'Blue',
        'ru': 'Синий'
    },
    'Blur' : {
        'en' : 'Blur',
        'ru' : 'Размытие',
    },
    'Border_type' : {
        'en' : 'Border type',
        'ru' : 'Тип границы',
    },
    
    'Bottom_to_top': {
        'en': 'Bottom to top',
        'ru': 'Снизу вверх',
    }, 
    'BW_mask': {
        'en': 'B/W mask',
        'ru': 'Ч/Б маска'
    },
    'Cancel': {
        'en': 'Cancel',
        'ru': 'Отмена'
    },
    'Center_at_cursor':{
        'en' : 'Center at the cursor',
        'ru' : 'Центрировать на курсоре',
    },
    'Change': {
        'en': 'Change',
        'ru': 'Изменить'
    },
    'Change_file_format': {
        'en': 'Change file format',
        'ru': 'Изменить формат файла'
    },
    'Channels_deform' : {
        'en': 'Channels deform',
        'ru': 'Деформация каналов'
    },
    'Clear': {
        'en': 'Clear',
        'ru': 'Очистить'
    },
    
    'Close': {
        'en': 'Close',
        'ru': 'Закрыть'
    },
    'Color': {
        'en': 'Color',
        'ru': 'Цвет',
    }, 
    'Computing': {
        'en': 'Computing',
        'ru': 'Вычисление'
    },
    'Confidence': {
        'en': 'Confidence',
        'ru': 'Вероятность'
    },
    'Confirm': {
        'en': 'Confirm',
        'ru': 'Подтвердить'
    },
    'Copy_image': {
        'en': 'Copy image',
        'ru': 'Скопировать изображение'
    },
    'Copy_mask': {
        'en': 'Copy mask',
        'ru': 'Скопировать маску'
    },
    'Current_config' : {
        'en' : 'Current config',
        'ru' : 'Текущая конфигурация',
    },
    'Cut': {
        'en': 'Cut',
        'ru': 'Вырезать'
    },
    'Cut_3p_bezier': {
        'en': 'Cut 3p bezier',
        'ru': 'Вырезать 3т безье'
    },
    'Cut_edges': {
        'en': 'Cut edges',
        'ru': 'Обрезка граней'
    },
    'Data_generator': {
        'en': 'Data generator',
        'ru': 'Генератор данных'
    },
    'Decrease_chance_similar' : {
        'en' : 'Decrease chance of similar',
        'ru' : 'Уменьшить шанс похожих',
    },
    'Decrease_opacity': {
        'en': 'Decrease opacity',
        'ru': 'Уменьшить непрозрачность'
    },
    'Default': {
        'en': 'Default',
        'ru': 'По-умолчанию'
    },
    'Image_deform_intensity' : {
        'en' : 'Image deform intensity',
        'ru' : 'Интенсивность деформации изображения',
    },
    'Target_image_deform_intensity' : {
        'en' : 'Target image deform intensity',
        'ru' : 'Интенсивность деформации целевого изображения',
    },
    'Dataset': {
        'en': 'Dataset',
        'ru': 'Датасет'
    },
    'Delete': {
        'en': 'Delete',
        'ru': 'Удалить'
    },
    'Delete_mask_and_file': {
        'en': 'Delete mask and file',
        'ru': 'Удалить маску и файл'
    },
    'Delete_export_directories': {
        'en': 'Delete export directories',
        'ru': 'Удалить директории экспорта'
    },
    'Delete_output_directory': {
        'en': 'Delete output directory',
        'ru': 'Удалить выходную директорию'
    },
    'Deleting_metadata': {
        'en': 'Deleting metadata',
        'ru': 'Удаление метаданных'
    },
    'Depth': {
        'en': 'Depth',
        'ru': 'Глубина'
    },
    'Device': {
        'en': 'Device',
        'ru': 'Устройство'
    },
    'Draw_annotations': {
        'en': 'Draw annotations',
        'ru': 'Рисовать аннотации'
    },    
    'Dimension': {
        'en': 'Dimension',
        'ru': 'Размерность'
    },
    'Directory':{
        'en' : 'Directory',
        'ru' : 'Директория',
    },
    'Disable_export': {
        'en': 'Disable export',
        'ru': 'Отключить экспорт'
    },
    'Discard_if_more': {
        'en': 'Discard if more',
        'ru': 'Отбросить если более'
    },
    'Dist_from_center': {
        'en': 'Dist from center',
        'ru': 'Расстояние от центра'
    },
    'Effective_iteration_time' : {
        'en' : 'Effective iteration time',
        'ru' : 'Эффективное время итерации',
    },
    'Enable_export': {
        'en': 'Enable export',
        'ru': 'Включить экспорт'
    },
    'Error': {
        'en': 'Error',
        'ru': 'Ошибка'
    },
    'Edit': {
        'en': 'Edit',
        'ru': 'Правка'
    },
    'Enhanced': {
        'en': 'Enhanced',
        'ru': 'Улучшенное',
    },
    'Enhancer': {
        'en': 'Enhancer',
        'ru': 'Улучшатель'
    },
    'End': {
        'en': 'End',
        'ru': 'Конец'
    },
    'Every': {
        'en': 'Every',
        'ru': 'Каждые'
    },
    'Export': {
        'en': 'Export',
        'ru': 'Экспорт'
    },
    'Export_DFL_mask': {
        'en': 'Export DFL mask',
        'ru': 'Экспорт DFL маски'
    },
    'Export_mesh_mask': {
        'en': 'Export mesh mask',
        'ru': 'Экспорт меш маски'
    },
    'Export_metadata': {
        'en': 'Export metadata',
        'ru': 'Экспорт метаданных'
    },
    'Face_aligner': {
        'en': 'Face aligner',
        'ru': 'Выравниватель лица'
    },
    'Face_coverage': {
        'en': 'Face coverage',
        'ru': 'Покрытие лица'
    },
    'Face_detector': {
        'en': 'Face detector',
        'ru': 'Детектор лица'
    },
    'Face_identifier': {
        'en': 'Face identifier',
        'ru': 'Идентификатор лица'
    },
    'Face_similarity_reference': {
        'en': 'Face similarity (reference)',
        'ru': 'Похожесть лица (по образцу)'
    },
    'Face_similarity_clustering': {
        'en': 'Face similarity (clustering)',
        'ru': 'Похожесть лица (кластеризация)'
    },
    'Face_list': {
        'en': 'Face list',
        'ru': 'Список лиц',
    },
    'Face_marker': {
        'en': 'Face marker',
        'ru': 'Маркер лица'
    },
    'Face_pitch': {
        'en': 'Face pitch',
        'ru': 'Наклон лица'
    },
    'Face_realign': {
        'en': 'Face realign',
        'ru': 'Перевыровнить лицо'
    },
    'Face_source_size': {
        'en': 'Face source size',
        'ru': 'Размер лица в исходнике'
    },
    'Face_yaw': {
        'en': 'Face yaw',
        'ru': 'Поворот лица'
    },
    'Face_Y_offset': {
        'en': 'Face Y offset',
        'ru': 'Смещение лица по Y'
    },
    'Face_Y_axis_offset': {
        'en': 'Face Y axis offset',
        'ru': 'Смещение Y оси'
    },
    'File': {
        'en': 'File',
        'ru': 'Файл'
    },
    'File_format': {
        'en': 'File format',
        'ru': 'Формат файла'
    },
    'File_state_manager': {
        'en': 'File state manager',
        'ru': 'Менеджер файлового состояния'
    },
    'Fill': {
        'en': 'Fill',
        'ru': 'Залить'
    },
    'Fill_3p_bezier': {
        'en': 'Fill 3p bezier',
        'ru': 'Залить 3т безье'
    },
    'Filter': {
        'en': 'Filter',
        'ru': 'Фильтр'
    },
    'Fix_borders' : {
        'en' : 'Fix borders',
        'ru' : 'Фикс границ',
    },
    'FImage.Border.CONSTANT' : {
        'en' : 'Constant',
        'ru' : 'Постоянный',
    },
    'FImage.Border.REFLECT' : {
        'en' : 'Reflect',
        'ru' : 'Отражённый',
    },
    'FImage.Border.REPLICATE' : {
        'en' : 'Replicate',
        'ru' : 'Скопированный',
    },
    'FImage.Interp.NEAREST' : {
        'en' : 'Nearest',
        'ru' : 'Ближайший',
    },
    'FImage.Interp.LINEAR' : {
        'en' : 'Linear',
        'ru' : 'Линейный',
    },
    'FImage.Interp.CUBIC' : {
        'en' : 'Cubic',
        'ru' : 'Кубический',
    },
    'FImage.Interp.LANCZOS4' : {
        'en' : 'Lanczos4',
        'ru' : 'Ланцош4',
    },
    'Flip' : {
        'en' : 'Flip',
        'ru' : 'Отразить',
    },
    'Gathering_metadata': {
        'en': 'Gathering metadata',
        'ru': 'Сбор метаданных'
    },
    'Generate': {
        'en': 'Generate',
        'ru': 'Генерация'
    },
    'Generate_preview': {
        'en': 'Generate preview',
        'ru': 'Генерация превью'
    },
    'Generalization_level' : {
        'en' : 'Generalization level',
        'ru' : 'Уровень обобщения',
    },
    'Generating': {
        'en': 'Generating',
        'ru': 'Генерация'
    },
    'Gray_image': {
        'en': 'Gray image',
        'ru': 'Серое изображение'
    },
    'Green': {
        'en': 'Green',
        'ru': 'Зелёный'
    },
    'Half_edge': {
        'en': 'Half edge',
        'ru': 'Уполовинить грань'
    },
    'Help':{
        'en' : 'Help',
        'ru' : 'Помощь',
    },
    'Histogram_similarity': {
        'en': 'Histogram similarity',
        'ru': 'Похожесть гистограммы'
    },
    'Hold': {
        'en': 'Hold',
        'ru': 'Удерживать'
    },
    'Increase_opacity': {
        'en': 'Increase opacity',
        'ru': 'Увеличить непрозрачность'
    },
    'Input_image': {
        'en': 'Input image',
        'ru': 'Входное изображение'
    },
    'Interpolation': {
        'en': 'Interpolation',
        'ru': 'Интерполяция'
    },
    'Image': {
        'en': 'Image',
        'ru': 'Изображение'
    },
    'Image_and_mask': {
        'en': 'Image and mask',
        'ru': 'Изображение и маска'
    },
    'Image_and_gray_image': {
        'en': 'Image and gray image',
        'ru': 'Изображение и серое изображение'
    },
    'Image_size': {
        'en': 'Image size',
        'ru': 'Размер изображения'
    },
    'Image_index' : {
        'en' : 'Image index',
        'ru' : 'Индекс изображения',
    },
    'Import': {
        'en': 'Import',
        'ru': 'Импорт'
    },
    'Import_metadata': {
        'en': 'Import metadata',
        'ru': 'Импорт метаданных'
    },
    'Info': {
        'en': 'Info',
        'ru': 'Инфо'
    },
    'Input_channel_type': {
        'en': 'Input channel type',
        'ru': 'Тип входного канала'
    },
    'Input_directory': {
        'en': 'Input directory',
        'ru': 'Входная директория'
    },
    'Invert_selection': {
        'en': 'Invert selection',
        'ru': 'Инвертировать выделенное'
    },
    'Items': {
        'en': 'Items',
        'ru': 'Элементы'
    },
    'Iteration_time' : {
        'en' : 'Iteration time',
        'ru' : 'Время итерации',
    },
    'Invalid_input_data_explained': {
        'en': 'Invalid input data. This can happen if the configuration of the model has changed when the training is enabled.',
        'ru': 'Неправильные входные данные. Это может случиться, если изменилась конфигурация модели при включенной тренировке.'
    },
    'Jobs' : {
        'en' : 'Jobs',
        'ru' : 'Задания',
    },
    'JPEG_artifacts' : {
        'en' : 'JPEG artifacts',
        'ru' : 'JPEG артефакты',
    },
    'Keep_view':{
        'en' : 'Keep view',
        'ru' : 'Сохранять вид',
    },
    'Language':{
        'en' : 'Language',
        'ru' : 'Язык',
    },
    'Largest': {
        'en': 'Largest',
        'ru': 'Наибольшему'
    },
    'Learning_rate' : {
        'en' : 'Learning rate',
        'ru' : 'Скорость обучения',
    },
    'Left_to_right': {
        'en': 'Left to right',
        'ru': 'Слева направо'
    },
    'Levels_range': {
        'en': 'Levels range',
        'ru': 'Диапазон уровней'
    },
    'Levels_shift' : {
        'en' : 'Levels shift',
        'ru' : 'Смещение уровней',
    },
    'Load': {
        'en': 'Load',
        'ru': 'Загрузить'
    },
    'Loading': {
        'en': 'Loading',
        'ru': 'Загрузка'
    },
    'Loading_meta_data': {
        'en': 'Loading meta data',
        'ru': 'Загрузка мета данных'
    },
    'Luminance': {
        'en': 'Luminance',
        'ru': 'Яркость',
    }, 
    'Marked':{
        'en' : 'Marked',
        'ru' : 'Помечено',
    },
    'Mask': {
        'en': 'Mask',
        'ru': 'Маска',
    },
    'Mask_from_landmarks': {
        'en': 'Mask from landmarks',
        'ru': 'Маску из лицевых точек'
    },
    'Max_faces': {
        'en': 'Max faces',
        'ru': 'Максимум лиц',
    },
    'Maximum': {
        'en': 'Maximum',
        'ru': 'Максимум'
    },
    'Mark_unmark_selected': {
        'en': 'Mark/unmark selected',
        'ru': 'Пометить/убрать выделенное'
    },
    'Mask_name':{
        'en' : 'Mask name',
        'ru' : 'Имя маски',
    },
    'Menu_unselected': {
        'en': '-- unselected --',
        'ru': '-- не выбрано --'
    },
    'Method': {
        'en': 'Method',
        'ru': 'Метод'
    },
    'Metadata': {
        'en': 'Metadata',
        'ru': 'Метаданные'
    },
    'Metrics' : {
        'en' : 'Metrics',
        'ru' : 'Метрики',
    },
    'Min_face_size': {
        'en': 'Min face size',
        'ru': 'Минимальный размер лица'
    },
    
    'Minimum_confidence': {
        'en': 'Minimum confidence',
        'ru': 'Минимальная вероятность'
    },
    
    'Mode': {
        'en': 'Mode',
        'ru': 'Режим'
    },
    

    'Mode.Fit' : {
        'en' : 'Fit',
        'ru' : 'Вместить',
    },

    'Mode.Patch' : {
        'en' : 'Patch',
        'ru' : 'Патч',
    },

    
    'Model': {
        'en': 'Model',
        'ru': 'Модель'
    },
    'Move_selected_to_directory': {
        'en': 'Move selected to a directory',
        'ru': 'Переместить выбранное в директорию'
    },
    'Moving': {
        'en': 'Moving',
        'ru': 'Перемещение'
    },
    'Name': {
        'en': 'Name',
        'ru': 'Имя'
    },
    'Navigation': {
        'en': 'Navigation',
        'ru': 'Навигация'
    },
    'Next': {
        'en': 'Next',
        'ru': 'Следующий'
    },
    'Network_type': {
        'en': 'Network type',
        'ru': 'Тип сети'
    },
    'New': {
        'en': 'New',
        'ru': 'Новый'
    },
    'No_image_selected':{
        'en' : 'No image selected',
        'ru' : 'Не выбрано изображение',
    },
    'No_landmark_metadata': {
        'en': 'No landmark metadata',
        'ru': 'Нет метаданных ландмарков'
    },
    'No_mask_selected':{
        'en' : 'No mask selected',
        'ru' : 'Не выбрана маска',
    },
    'No_meta_data':{
        'en' : 'No meta data',
        'ru' : 'Нет мета данных',
    },
    
    'Not_applied': {
        'en': 'Not applied',
        'ru': 'Не применено'
    },
    'Offset' : {
        'en' : 'Offset',
        'ru' : 'Смещение',
    },
    'Ok': {
        'en': 'Ok',
        'ru': 'Ок'
    },
    'Opacity': {
        'en': 'Opacity',
        'ru': 'Прозрачность'
    },
    'Open': {
        'en': 'Open',
        'ru': 'Открыть'
    },
    'Overlap_threshold': {
        'en': 'Overlap threshold',
        'ru': 'Порог перекрытия'
    },
    'Output_channel_type': {
        'en': 'Output channel type',
        'ru': 'Тип выходного канала'
    },
    'Overlay': {
        'en': 'Overlay',
        'ru': 'Оверлей'
    },
    'Output_directory': {
        'en': 'Output directory',
        'ru': 'Выходная директория'
    },
    'Output_type': {
        'en': 'Output type',
        'ru': 'Выходной тип'
    },
    'Paired_image': {
        'en': 'Paired image',
        'ru': 'Парное изображение'
    },
    'Pair_type': {
        'en': 'Pair type',
        'ru': 'Тип пары'
    },
    'Pass_count': {
        'en': 'Pass count',
        'ru': 'Число проходов'
    },
    'Paste_mask': {
        'en': 'Paste mask',
        'ru': 'Вставить маску'
    },
    'Patch_mode' : {
        'en' : 'Patch mode',
        'ru' : 'Режим патча',
    },
    'Patch_size' : {
        'en' : 'Patch size',
        'ru' : 'Размер патча',
    },
    'Perceptual_dissimilarity': {
        'en': 'Perceptual dissimilarity',
        'ru': 'Непохожесть по восприятию'
    },
    'Perpendicular_constraint': {
        'en': 'Perpendicular constraint',
        'ru': 'Ограничение по перпендикуляру'
    },
    
    'Polygon': {
        'en': 'Polygon',
        'ru': 'Полигон'
    },
    'Predicted_image' : {
        'en' : 'Predicted image',
        'ru' : 'Предсказанное изображение',
    },
    'Predicted_guide' : {
        'en' : 'Predicted guide',
        'ru' : 'Предсказанный гайд',
    },
    'Predicted_mask' : {
        'en' : 'Predicted mask',
        'ru' : 'Предсказанная маска',
    },
    'Previous': {
        'en': 'Previous',
        'ru': 'Предыдущий'
    },
    'Preparing': {
        'en': 'Preparing',
        'ru': 'Подготовка'
    },
    'Preview': {
        'en': 'Preview',
        'ru': 'Предпросмотр'
    },
    'Processing':{
        'en' : 'Processing',
        'ru' : 'Обработка',
    },
    'Process_priority':{
        'en' : 'Processs priority',
        'ru' : 'Приоритет процесса',
    },
    'Process_priority.Normal':{
        'en' : 'Normal',
        'ru' : 'Нормальный',
    },
    'Process_priority.Low':{
        'en' : 'Low',
        'ru' : 'Низкий',
    },
    
    'Profile': {
        'en': 'Profile',
        'ru': 'Профиль'
    },
    'Purple': {
        'en': 'Purple',
        'ru': 'Розовый'
    },
    'Quality': {
        'en': 'Quality',
        'ru': 'Качество'
    },
    'Quit':{
        'en' : 'Quit',
        'ru' : 'Выход',
    },
    'Random' : {
        'en' : 'Random',
        'ru' : 'Случайно',
    },
    'Random_augmentations' : {
        'en' : 'Random augmentations',
        'ru' : 'Случайные аугментации',
    },
    'Random_transformations' : {
        'en' : 'Random transformations',
        'ru' : 'Случайные трансформации',
    },
    'Rate': {
        'en': 'Rate',
        'ru': 'Шанс'
    },
    'Reconstruction': {
        'en': 'Reconstruction',
        'ru': 'Реконструкция'
    },
    'Red': {
        'en': 'Red',
        'ru': 'Красный'
    },
    'Redo': {
        'en': 'Redo',
        'ru': 'Повторить'
    },
    'Reload': {
        'en': 'Reload',
        'ru': 'Перезагрузить'
    },
    'Remove_item': {
        'en': 'Remove item',
        'ru': 'Убрать позицию'
    },
    'Renaming': {
        'en': 'Renaming',
        'ru': 'Переименовывание'
    },
    'Reset': {
        'en': 'Reset',
        'ru': 'Сброс'
    },
    'Reset_model' : {
        'en' : 'Reset model',
        'ru' : 'Сбросить модель',
    },
    'Reset_UI':{
        'en' : 'Reset UI',
        'ru' : 'Сброс интерфейса',
    },
    'Resize' : {
        'en' : 'Resize',
        'ru' : 'Пережатие',
    },
    'Resolution': {
        'en': 'Resolution',
        'ru': 'Разрешение'
    },
    'Reveal_in_explorer': {
        'en': 'Reveal in explorer',
        'ru': 'Открыть в проводнике'
    },
    'Right_to_left': {
        'en': 'Right to left',
        'ru': 'Справа налево'
    },
    'Rotation' : {
        'en' : 'Rotation',
        'ru' : 'Поворот',
    },
    'Sample_count' : {
        'en' : 'Sample count',
        'ru' : 'Кол-во семплов',
    },
    'Save': {
        'en': 'Save',
        'ru': 'Сохранить'
    },
    'Segmentator': {
        'en': 'Segmentator',
        'ru': 'Сегментатор'
    },
    'Scale' : {
        'en' : 'Scale',
        'ru' : 'Масштаб',
    },
    'Selected': {
        'en': 'Selected',
        'ru': 'Выделено'
    },
    'Select_marked': {
        'en': 'Select marked',
        'ru': 'Выделить помеченное'
    },
    'Select_unselect_all': {
        'en': 'Select/unselect all',
        'ru': 'Выбрать/убрать все'
    },
    'Sharpen' : {
        'en' : 'Sharpen',
        'ru' : 'Резкость',
    },
    'Show_file_name': {
        'en': 'Show file name',
        'ru': 'Показывать имя файла'
    },
    'Show_file_extension': {
        'en': 'Show file extension',
        'ru': 'Показывать расширение файла'
    },
    'Show_face_landmarks': {
        'en': 'Show face landmarks',
        'ru': 'Показывать лицевые точки'
    },
    'Show_lines_size': {
        'en': 'Show lines size',
        'ru': 'Показать размеры линий'
    },
    'Show_hide_console':{
        'en' : 'Show/hide console',
        'ru' : 'Показать/скрыть консоль',
    },
    'Smooth_corner': {
        'en': 'Smooth corner',
        'ru': 'Сгладить угол'
    },
    'Sort': {
        'en': 'Sort',
        'ru': 'Сортировка'
    },
    'Sorting': {
        'en': 'Sorting',
        'ru': 'Сортировка'
    },
    'Sort_by': {
        'en': 'Sort by',
        'ru': 'Отсортировать по'
    },
    'Source': {
        'en': 'Source',
        'ru': 'Источник'
    },
    'Source_type': {
        'en': 'Source type',
        'ru': 'Тип источника'
    },
    'Source_sequence_number': {
        'en': 'Source sequence number',
        'ru': 'Исходный порядковый номер'
    },
    'Stage' : {
        'en' : 'Stage',
        'ru' : 'Стадия',
    },
    'Start': {
        'en': 'Start',
        'ru': 'Начать'
    },
    'Start_training': {
        'en': 'Start training',
        'ru': 'Начать тренировку'
    },
    'State': {
        'en': 'State',
        'ru': 'Состояние'
    },
    'State_name': {
        'en': 'State name',
        'ru': 'Имя состояния'
    },
    'Static_augmentations' : {
        'en' : 'Static augmentations',
        'ru' : 'Статичные аугментации',
    },
    'Step': {
        'en': 'Step',
        'ru': 'Шаг'
    },
    'Stop_training': {
        'en': 'Stop training',
        'ru': 'Остановить тренировку'
    },
    'Success': {
        'en': 'Success',
        'ru': 'Успешно'
    },   
    'Swap': {
        'en': 'Swap',
        'ru': 'Замена'
    },
    'Swapper': {
        'en': 'Swapper',
        'ru': 'Заменитель'
    },
    'Swap_enhanced': {
        'en': 'Swap enhanced',
        'ru': 'Улучшенная замена'
    },
    'System': {
        'en': 'System',
        'ru': 'Система'
    },  
    'Files_of_selected_mask_will_be_overwritten': {
        'en': 'The files of the selected mask will be overwritten.',
        'ru': 'Файлы выбранной маски будут перезаписаны.'
    },
    'Target': {
        'en': 'Target',
        'ru': 'Цель',
    },
    'Target_image': {
        'en': 'Target image',
        'ru': 'Целевое изображение',
    },
    'Target_guide' : {
        'en' : 'Target guide',
        'ru' : 'Целевой гайд',
    },
    'Target_mask' : {
        'en' : 'Target mask',
        'ru' : 'Целевая маска',
    },
    'Thumbnail_size': {
        'en': 'Thumbnail size',
        'ru': 'Размер эскиза'
    },
    'Toggle': {
        'en': 'Toggle',
        'ru': 'Сменить',
    }, 
    'Top_to_bottom': {
        'en': 'Top to bottom',
        'ru': 'Сверху вниз',
    }, 
    'Total': {
        'en': 'Total',
        'ru': 'Всего',
    },
    'Train': {
        'en': 'Train',
        'ru': 'Тренировать'
    },
    'Trainer': {
        'en': 'Trainer',
        'ru': 'Тренер'
    },
    'Training_error': {
        'en': 'Training error',
        'ru': 'Ошибка тренировки'
    },
    'Transform': {
        'en': 'Transform',
        'ru': 'Трансформация'
    },
    'Transform_intensity' : {
        'en' : 'Transform intensity',
        'ru' : 'Интенсивность транформации',
    },
    'Translation_X' : {
        'en' : 'Translation-X',
        'ru' : 'Перенос по X',
    },
    'Translation_Y' : {
        'en' : 'Translation-Y',
        'ru' : 'Перенос по Y',
    },
    'Trash_directory': {
        'en': 'Trash directory',
        'ru': 'Директория для мусора'
    },
    'Trash_selected': {
        'en': 'Trash selected',
        'ru': 'Выбросить выделенное'
    },
    'Type': {
        'en': 'Type',
        'ru': 'Тип',
    },
    'Warning': {
        'en': "Warning",
        'ru': "Предупреждение"
    },
    'Pair_type_files_will_be_deleted': {
        'en': "Pair-type files will be deleted.",
        'ru': "Файлы типа пар будут удалены."
    },
    'Undo': {
        'en': 'Undo',
        'ru': 'Отменить'
    },
    'Filter_by_uniform_distribution': {
        'en': 'Filter by uniform distribution',
        'ru': 'Фильтр по равномерному распределению'
    },
    'View': {
        'en': 'View',
        'ru': 'Вид'
    },
    'Video_file': {
        'en': 'Video file',
        'ru': 'Видео файл'
    },
    'Image_sequence': {
        'en': 'Image sequence',
        'ru': 'Ряд изображений'
    },
    
    # Lower case
    'and_next': {
        'en': 'and next',
        'ru': 'и следующий'
    },
    'and_previous': {
        'en': 'and previous',
        'ru': 'и предыдущий'
    },
    'backups': {
        'en': 'backups',
        'ru': 'копий'
    },
    'by_pair_type': {
        'en': 'by pair type',
        'ru': 'по типу пары'
    },
    'by_name': {
        'en': 'by name',
        'ru': 'по имени'
    },
    'error': {
        'en': 'error',
        'ru': 'ошибка'
    },
    
    'dimension': {
        'en': 'dimension',
        'ru': 'размерность'
    },
    'it_s': {
        'en': 'it/s',
        'ru': 'ит/c'
    },
    'minutes': {
        'en': 'minutes',
        'ru': 'минут'
    },
    'no_items': {
        'en': 'no items',
        'ru': 'нет элементов'
    },
    'no_pair': {
        'en': 'no pair',
        'ru': 'нет пары'
    },
    'open_the_source': {
        'en': 'open the source',
        'ru': 'откройте источник'
    },
    'open_the_dataset': {
        'en': 'open the dataset',
        'ru': 'откройте датасет'
    },
    'per_second': {
        'en': 'per second',
        'ru': 'в секунду'
    },
    'second' : {
        'en' : 'second',
        'ru' : 'секунд',
    },
    'with_mask': {
        'en': 'with mask',
        'ru': 'с маской'
    },
    



    
    
    
    
    
    # 'QxDataGenerator.Cut_edges' : {
    #     'en' : 'Cut edges',
    #     'ru' : 'Обрезка граней',
    # },


    

    # 'QxDataGenerator.Glow_shade' : {
    #     'en' : 'Glow/shade',
    #     'ru' : 'Блики/тени',
    # },

    

  

    # 'QxDataGenerator.Output_type' : {
    #     'en' : 'Output type',
    #     'ru' : 'Выходной тип',
    # },
    
    # 'QxDataGenerator.Image_n_Mask' : {
    #     'en' : 'Image and mask',
    #     'ru' : 'Изображение и маска',
    # },
    
    # 'QxDataGenerator.Image_n_ImageGrayscaled' : {
    #     'en' : 'Image and Image grayscaled',
    #     'ru' : 'Изображение и обесцвеченное изображение',
    # },
    
    # 'QxDataGenerator.Generate_preview' : {
    #     'en' : 'Generate preview',
    #     'ru' : 'Генерировать предпросмотр',
    # },

    # 'QxDataGenerator.Image' : {
    #     'en' : 'Image',
    #     'ru' : 'Изображение',
    # },

    # 'QxDataGenerator.Mask' : {
    #     'en' : 'Mask',
    #     'ru' : 'Маска',
    # },
    
    # 'MxModel.Exporting_model_to' : {
    #     'en' : 'Exporting model to',
    #     'ru' : 'Экспортируем модель в',
    # },
    
    # 'MxModel.Importing_model_from' : {
    #     'en' : 'Importing model from',
    #     'ru' : 'Импортируем модель из',
    # },
    
    # 'MxModel.Downloading_model_from' : {
    #     'en' : 'Downloading model from',
    #     'ru' : 'Скачиваем модель из',
    # },
    
    # 'QxModel.Device' : {
    #     'en' : 'Device',
    #     'ru' : 'Устройство',
    # },


   
    
    

    # 'QxModel.UNet_mode' : {
    #     'en' : 'U-Net mode',
    #     'ru' : 'U-Net режим',
    # },
    
    # 'QxModel.Input' : {
    #     'en' : 'Input',
    #     'ru' : 'Вход',
    # },

    # 'QxModel.InputType.Color' : {
    #     'en' : 'Color',
    #     'ru' : 'Цвет',
    # },

    # 'QxModel.InputType.Luminance' : {
    #     'en' : 'Luminance',
    #     'ru' : 'Яркость',
    # },

    

    
    
    
    
    # 'QxModel.Import_model' : {
    #     'en' : 'Import model',
    #     'ru' : 'Импорт модели',
    # },
    
    # 'QxModel.Export_model' : {
    #     'en' : 'Export model',
    #     'ru' : 'Экспорт модели',
    # },
    
    # 'QxModel.Download_pretrained_model' : {
    #     'en' : 'Download pretrained model',
    #     'ru' : 'Скачать предтренированную модель',
    # },
    
    
    
    

    # 'QxModelTrainer.power' : {
    #     'en' : 'power',
    #     'ru' : 'сила',
    # },

    
    
    
    
    
    

    

    # 'QxExport.Input' : {
    #     'en' : 'Input',
    #     'ru' : 'Вход',
    # },
    
    # 'QxExport.Output' : {
    #     'en' : 'Output',
    #     'ru' : 'Выход',
    # },
    
    # 'QxExport.Output_image' : {
    #     'en' : 'Output image',
    #     'ru' : 'Выход изображения',
    # },
    
    # 'QxExport.Output_mask' : {
    #     'en' : 'Output mask',
    #     'ru' : 'Выход маски',
    # },
    
    

    # 'QxExport.Sample_count' : {
    #     'en' : 'Sample count',
    #     'ru' : 'Кол-во семплов',
    # },
    
    # 'QxExport.Fix_borders' : {
    #     'en' : 'Fix borders',
    #     'ru' : 'Фикс границ',
    # },
    
    # 'QxPreview.Source' : {
    #     'en' : 'Source',
    #     'ru' : 'Источник',
    # },

    # 'QxPreview.Data_generator' : {
    #     'en' : 'Data generator',
    #     'ru' : 'Генератор данных',
    # },

    

    # 'QxPreview.Generate':{
    #     'en' : 'Generate',
    #     'ru' : 'Генерировать',
    # },

    

    

    
    
    
    
    
    
    

    
    

    
    
    

  

}




pat = re.compile(r'@\([^\)]*\)|[^@$]+')
def L(s : str|None, lang : str) -> str|None:
    """
    Localize string.

    @(id)...@(id2)...
    """
    if s is None:
        return None
    
    result = []
    for sub in pat.findall(s):
        if sub[:2] == '@(' and sub[-1] == ')':
            if (l := _id_to_string_dict.get(sub[2:-1], None) ) is not None:
                if (ll := l.get(lang, None)) is None:
                    ll = l['en']

                result.append(ll)
            else:
                result.append(sub)
                print(f'No localization found for: {sub}')
        else:
            result.append(sub)
    
    return ''.join(result)
