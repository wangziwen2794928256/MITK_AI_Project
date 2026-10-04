#ifndef AISEGMENTATIONACTIVATOR_H
#define AISEGMENTATIONACTIVATOR_H


#include <QObject>
#include <ctkPluginActivator.h>


class AIsegmentationActivator :

        public QObject,
        public ctkPluginActivator

{

Q_OBJECT

Q_PLUGIN_METADATA(IID
"org_mitk_gui_qt_ai_segmentation")


Q_INTERFACES(ctkPluginActivator)


public:


void start(ctkPluginContext* context) override;


void stop(ctkPluginContext* context) override;


};


#endif