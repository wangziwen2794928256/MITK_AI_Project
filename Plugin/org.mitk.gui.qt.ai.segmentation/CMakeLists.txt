project(org_mitk_gui_qt_ai_segmentation)


mitk_create_plugin(
    EXPORT_DIRECTIVE AISEGMENTATION_EXPORT
    EXPORTED_INCLUDE_SUFFIXES src
    MODULE_DEPENDS
        MitkQtWidgetsExt
        MitkCore
)