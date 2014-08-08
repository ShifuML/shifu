To install this plugin:

mvn install
cd target
tar zxvf shifu-plugin-pig-1.0-SNAPSHOT-release.tar.gz-release.tar.gz
cd shifu-plugin-pig-1.0-SNAPSHOT
cp -r * $SHIFU_HOME/plugin/


Example request.json using this plugin:

{
    "name": "ModelExec",
    "description": "step 7, execute model",
    "processor": {
        "spi": "RequestProcessor",
        "impl": "ml.shifu.plugin.pig.PigModelExecRequestProcessor",
        "params": {
            "modelName": "demoModel",
            "pathPMML": "wdbcDataSet/model_output.xml",
            "pathData": "wdbcDataSet/wdbc.eval",
            "pathHeader": "wdbcDataSet/wdbc.header",
            "pathResult": "wdbcDataSet/generated/",
            "delimiter": ",",
            "headerDelimiter": "|",
            "localMode": false
        }
    },
    "bindings": []
}

