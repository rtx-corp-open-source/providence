// **Raytheon Technologies proprietary**
// Export controlled - see license file
library identifier: 'DSCI-jenkins-pipeline-utilities@master', retriever: modernSCM(
  [$class: 'GitSCMSource',
   remote: 'https://github.devops.utc.com/Type-C-Lite/RTXDS-jenkins-pipeline-utilities.git',
   credentialsId: 'CORP-GitHub-DSCI-App-Connector'])
   
aimlCardsPipeline {
    sonarProjectKey="RTXDS-providence"
    sonarProjectName="RTXDS providence"
    sonarSourcePath="providence"
    sonarProjectVersion="1.0"
    libVersionFilePath="providence/_version.py"

    //don't change anything below, unless you know what are you changing. These are Azure Web app environment deployment details
    acrWebAppName="ds-providence-website"
    xetaAzureDevWebAppName="x12d3rtxedx-linweb-aiml-providence"
    //xetaAzureQAWebAppName=""
    xetaAzureProdWebAppName="x12p2rtxedx-linweb-aiml-providence"

}

