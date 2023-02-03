// **Raytheon Technologies proprietary**
// Export controlled - see license file

HTTP_PROXY = "http://devproxy.utc.com:80"
HTTPS_PROXY = "http://devproxy.utc.com:80"
NO_PROXY="localhost,127.0.0.0/8,10.0.0.0/8,artifactory.dx.utc.com,github.dx.utc.com,/var/run/docker.sock"
deployment_env = 'dev'
JB_NAME = '';
BUILD_ARTIFACT_PATH = ''
BUILD_VERSION = ''
BRANCH_NAME = ''
APP_VERSION = ''

buildBadge = addEmbeddableBadgeConfiguration(id: "build", subject: "Build")
checkmarxBadge = addEmbeddableBadgeConfiguration(id: "checkmarx", subject: "Checkmarx Scan")

pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile-jenkins'
            additionalBuildArgs """ --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg NO_PROXY=${NO_PROXY} """
            args """-u root --privileged \
            -v /var/lib/jenkins/.docker:/root/.docker \
            -v /datadisk/jenkins/tools:/datadisk/jenkins/tools \
            -v /var/lib/jenkins/.gitconfig:/root/.gitconfig \
            -v /var/lib/jenkins/.ssh:/root/.ssh \
            -e "HTTP_PROXY=${HTTP_PROXY}" -e "HTTPS_PROXY=${HTTPS_PROXY}" -e "NO_PROXY=${NO_PROXY}" \
            --entrypoint=\
            --security-opt seccomp=unconfined \
            """
            label 'ds-build-agent'
        }
    }
    options {
        disableConcurrentBuilds()
        timeout(time: 3, unit: 'HOURS')
    }
    environment {
        CI = 'true'
        HTTP_PROXY = "${HTTP_PROXY}"
        HTTPS_PROXY = "${HTTPS_PROXY}"
        NO_PROXY= "${NO_PROXY}"
        EMAIL_FROM= "datascience-jenkins@utc.com"
        BUILD_TIMESTAMP = sh(script: "echo `date +'%d%m%Y%H%M%S'`", returnStdout: true).trim()
        RELEASE_TEAM_APPROVER = "mcginnwi,prabhuch"
        RELEASE_TEAM_EMAIL = "william.mcginnis@utc.com,chetan.prabhu@utc.com"
        PROD_TEAM_APPROVER = "mcginnwi,prabhuch"
        PROD_TEAM_EMAIL = "william.mcginnis@utc.com,chetan.prabhu@utc.com"
        DEVOPS_TEAM_APPROVER = "krishnpr"
        DEVOPS_TEAM_EMAIL = "prabhatkumar.krishna@utc.com"
    }
    stages {
        stage('Pre-Initialize') {
            steps {
                declareEnvVariables()
                sendBuildInvokeInformation()
                createFolders()
            }
        }

        stage('Checkmarx scan') {
            steps {
                codeScanCheckmarx()
            }
        }
        
        stage('Unit Test') {
            steps {
                appTest()
            }
        }

        stage('SonarQube') {
            steps {
                codeScanSonarQube()
            }
        }

        stage('Approval for Release') {
            steps {
                deploymentConfirmation()
            }
        }

        stage('Publish') {
            steps {
                eggPublish()
            }
        }

        stage('Update GH Pages') {
            steps {
                updateGHPages()
            }
        }
    }
    post {
        always {
            executePostBuildJobs()
            sendBuildStatusInformation()
            echo 'Clean up workspace ..'
            cleanWs()
            echo 'Workspace Cleanup done..'
        }
        success {
            setBadgeStatus(buildBadge, "passing")
        }

        unstable{
            setBadgeStatus(buildBadge, "failing")    
        }

        failure {
            setBadgeStatus(buildBadge, "failing")
        }
    }
}

/**
 Declare and initialize variables 
*/
def declareEnvVariables() {
    BRANCH_NAME = sh(returnStdout: true, script: "echo $GIT_BRANCH").trim()
    echo "Branch Name : ${BRANCH_NAME}"
    
    APP_VERSION = sh(script: "python3 setup.py --version", returnStdout: true).trim()

    if(BRANCH_NAME.startsWith('release')) {
        deployment_env = 'rc'
        BUILD_VERSION= "${APP_VERSION}.rc${BUILD_NUMBER}"
    } else if(BRANCH_NAME =='master') {
        deployment_env = 'prod'
        BUILD_VERSION= APP_VERSION
    } else if(BRANCH_NAME.contains('develop')) {
        deployment_env = 'dev'
        BUILD_VERSION= "${APP_VERSION}.dev${BUILD_NUMBER}"
    } else {
        deployment_env ='skip'
    }

    JB_NAME = JOB_NAME.replaceAll("\\/","|")

    echo "APP_VERSION: ${APP_VERSION} BUILD VERSION : ${BUILD_VERSION}"
    
    BUILD_ARTIFACT_PATH = "results/build_artifacts_${deployment_env}_${BUILD_TIMESTAMP}"
}

/**
 Create folders for storing reports and artifacts files 
*/
def createFolders() {
   sh "mkdir -p ${BUILD_ARTIFACT_PATH}"
}

/**
 Scan the code with Checkmarx
 */
def codeScanCheckmarx() {
    if(deployment_env !='skip') {
        try{
            echo '........... Running CheckMarx Scan for Providence ...........'
            step([$class: 'CxScanBuilder', comment: '', credentialsId: '', dependencyScanConfig: [dependencyScanExcludeFolders: '', dependencyScanPatterns: '', 
                osaArchiveIncludePatterns: '*.zip, *.war, *.ear, *.tgz', scaAccessControlUrl: 'https://platform.checkmarx.net', scaCredentialsId: '', scaServerUrl: 'https://api-sca.checkmarx.net', scaTenant: '', 
                scaWebAppUrl: 'https://sca.checkmarx.net'], enableProjectPolicyEnforcement: true, excludeFolders: '.sonarscanner', exclusionsSetting: 'job', failBuildOnNewResults: false, 
                filterPattern: '''!**/_cvs/**/*, !**/.svn/**/*,   !**/.hg/**/*,   !**/.git/**/*,  !**/.bzr/**/*, !**/bin/**/*, !**/*.md, !**/*.txt,!**/*.in,!**/*Jenkinsfile*, !**/*jenkins*, !**/*Docker*,
                !**/obj/**/*,  !**/backup/**/*, !**/.idea/**/*, !**/*.DS_Store, !**/*.ipr,     !**/*.iws,
                !**/*.bak,     !**/*.tmp,       !**/*.aac,      !**/*.aif,      !**/*.iff,     !**/*.m3u, !**/*.mid, !**/*.mp3,
                !**/*.mpa,     !**/*.ra,        !**/*.wav,      !**/*.wma,      !**/*.3g2,     !**/*.3gp, !**/*.asf, !**/*.asx,
                !**/*.avi,     !**/*.flv,       !**/*.mov,      !**/*.mp4,      !**/*.mpg,     !**/*.rm,  !**/*.swf, !**/*.vob,
                !**/*.wmv,     !**/*.bmp,       !**/*.gif,      !**/*.jpg,      !**/*.png,     !**/*.psd, !**/*.tif, !**/*.swf,
                !**/*.jar,     !**/*.zip,       !**/*.rar,      !**/*.exe,      !**/*.dll,     !**/*.pdb, !**/*.7z,  !**/*.gz,
                !**/*.tar.gz,  !**/*.tar,       !**/*.gz,       !**/*.ahtm,     !**/*.ahtml,   !**/*.fhtml, !**/*.hdm,
                !**/*.hdml,    !**/*.hsql,      !**/*.ht,       !**/*.hta,      !**/*.htc,     !**/*.htd, !**/*.war, !**/*.ear,
                !**/*.htmls,   !**/*.ihtml,     !**/*.mht,      !**/*.mhtm,     !**/*.mhtml,   !**/*.ssi, !**/*.stm,
                !**/*.stml,    !**/*.ttml,      !**/*.txn,      !**/*.xhtm,     !**/*.xhtml,   !**/*.class, !**/*.iml, !Checkmarx/Reports/*.*''', fullScanCycle: 10, generatePdfReport: true, 
                groupId: '2a07faca-4c96-4c1c-8ac5-ec65a1465631', incremental: true, password: '{AQAAABAAAAAQaTiozVG5aQs3vpjMQfhh/MXyViGvzzzpsi98Ekc4fqk=}', preset: '36', 
                projectName: 'data-science-providence', sastEnabled: true, serverUrl: 'https://checkmarx.dx.utc.com', sourceEncoding: '1', username: ''])
        
            sh "cp Checkmarx/Reports/*.pdf ${BUILD_ARTIFACT_PATH}/  || exit 0"
            setBadgeStatus(checkmarxBadge, "passing")
        }catch(error){
            echo error.getMessage()
            echo "Error while checkmarx scans."
            setBadgeStatus(checkmarxBadge, "failing")
        }
    }        
}

/**
 Build node application and perform unit testing 
*/
def appTest() {
    try{
        sh 'make UNIT_TEST'
    }catch(error){
        echo error.getMessage()
        echo "Error running unit tests."
    }

    publishHTML (target : [allowMissing: true,
        alwaysLinkToLastBuild: true,
        keepAll: true,
        reportDir: 'htmlcov',
        reportFiles: 'index.html',
        reportName: 'Code Coverage',
        reportTitles: 'Code Coverage Report']
    )

    try{
        publishHTML (target : [allowMissing: true,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'report/toxtestenv',
            reportFiles: 'report.html',
            reportName: 'Tox Report',
            reportTitles: 'Tox Detailed Report']
        )
    }catch(error){
        echo error.getMessage()
        echo "Error while Publishing the Tox Report."
    }

    def testResult=junit 'test-report/pytest-report.xml'
    step([$class: 'CoberturaPublisher', coberturaReportFile: 'coverage.xml'])
}

/**
 Scan the code with SonarQube
 */
def codeScanSonarQube() {
    if(deployment_env !='skip') {
        withSonarQubeEnv('SonarQube') {
            sh "sonar-scanner"
        }
    }
}

/**
 Based on environment, send deployment confirmation alert and wait for approval 
*/
def deploymentConfirmation() {
    if(deployment_env =='rc') {
        sendBuildApproverAlert(RELEASE_TEAM_APPROVER)
        sendMailandConfirmfromTeam(RELEASE_TEAM_EMAIL,deployment_env)
        confirmFromUser(RELEASE_TEAM_APPROVER,"Release Candidate")
    } else if(deployment_env =='prod') {
        sendBuildApproverAlert(PROD_TEAM_APPROVER)
        sendMailandConfirmfromTeam(PROD_TEAM_EMAIL,deployment_env)
        confirmFromUser(PROD_TEAM_APPROVER,"Production")
    }
}

/**
 Get approval from approvers, defined based on deployment enviornment 
*/
def confirmFromUser(approvers,rmsg) {
    def userInput = input(
                            id: 'userInput', message: "This is ${rmsg} Deployment!", parameters: [
                            [$class: 'BooleanParameterDefinition', defaultValue: false, description: '', name: "Please confirm you sure to proceed and publish the wheel as ${rmsg}."]
                        ],
                        submitterParameter: 'submitter', submitter:approvers)

    if(!userInput) {
        error "Build wasn't confirmed"
    }
}



/**
 Perform after build job, cleanup, alert and job status 
*/
def executePostBuildJobs() {
    try {
        sh "chmod -R a+w \$PWD"
        //archiveBuildArtifacts()
        //sendEmailBuildArtifacts()
    } catch(error) {
        echo error.getMessage()
        echo "Error while running post build job."
    }
}

/**
 Create archive folder for Build Artifacts - reports, logs 
*/
def archiveBuildArtifacts() {
    try {
        echo "archiving build artifacts : ${BUILD_ARTIFACT_PATH}"
        //modify the result folder permission
        sh "chown -R jenkins:jenkins results || exit 0"
        zip zipFile:"${BUILD_ARTIFACT_PATH}.zip", archive: true, dir: "${BUILD_ARTIFACT_PATH}"
    } catch(error) {
        echo error.getMessage()
        echo "Error while archiving build job."
    }
}

/**
 Send mail with attachment to the DL 
*/
def sendEmailBuildArtifacts() {
    try {
        emailext attachLog: false, attachmentsPattern: 'results/**/*.zip',
        body: "${currentBuild.currentResult}: Job ${env.JOB_NAME} build ${env.BUILD_NUMBER}\n More info at: ${env.RUN_DISPLAY_URL}",
        from:"${EMAIL_FROM}",
        to: 'datasciencedx.automation@utc.com',
        subject: "Jenkins Build ${currentBuild.currentResult}: Job ${env.JOB_NAME}"
    } catch(error) {
        echo error.getMessage()
        echo "Error while sending mail."
    }
}

/**
 Send Build Invokation slack alerts 
*/
def sendBuildInvokeInformation() {
    def msg = """
                {
                    "attachments": [
                        {
                            "title": "Data Science Providence build triggered!",
                            "title_link": "${env.RUN_DISPLAY_URL}",
                            "text": "Build Triggred, Job: ${env.JOB_NAME} # ${BUILD_NUMBER}",
                            "color": "#FFC107",
                            "fields": [
                                {
                                    "title": "Status",
                                    "value": "Initiated"
                                },
                                {
                                    "title": "Branch",
                                    "value": "${BRANCH_NAME}"
                                }
                            ]
                        }
                    ]
                }
            """
    sendSlack(msg)
}

/**
 Send Slack alert about build status 
*/
def sendBuildStatusInformation() {
    def color = '#7CD197'
    def build_result=currentBuild.currentResult
    if(build_result == 'FAILURE' || build_result == 'ABORTED')
        color = '#FF0000'
        
    def msg = """
                {
                    "attachments": [
                        {
                            "title": "Data Science Providence Build Information",
                            "title_link": "${env.RUN_DISPLAY_URL}",
                            "text": "Job ${env.JOB_NAME} # ${BUILD_NUMBER}",
                            "color": "${color}",
                            "fields": [
                                {
                                    "title": "Status",
                                    "value": "${build_result}"
                                },
                                {
                                    "title": "Branch",
                                    "value": "${BRANCH_NAME}"
                                },
                                {
                                    "title": "App Version",
                                    "value": "${APP_VERSION}"
                                }
                            ]
                        }
                    ]
                }
            """
    sendSlack(msg)
}

/**
 Method for sending Slack alert 
*/
def sendSlack(message) {
    def response = httpRequest url:'https://hooks.slack.com/services/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',
                                httpMode:'POST', contentType: 'APPLICATION_JSON',
                                requestBody: "${message}",httpProxy:"${HTTP_PROXY}"
        println("Slack Message Status: "+response.status)
        println("Content: "+response.content)
    
}

/**
 Send mail for confirmation to the approvers 
*/
def sendMailandConfirmfromTeam(emailAddress,renv) {
    try {
        emailext attachLog: false,
        body: htmlMessage(renv),
        mimeType: 'text/html',
        from:"${EMAIL_FROM}",
        to: "${emailAddress}",
        subject: "Jenkins Build Awaiting Response: Job ${env.JOB_NAME}"
    } catch(error) {
        echo error.getMessage()
        echo "Error while sending mail."
    }
}

/**
 Mail template in html 
*/
def htmlMessage(renv) {
    def msg = """
        <html>
            <head>
                <style>
                    body{
                        padding:10px;
                        margin:10px;
                    }
                </style>
            </head>
            <body>
                <div>
                    <p>
                        Dear Team,

                        <br/>
                        <br/>
                        <h3> Environment : ${renv}</h3>
                    <p>
                    <p>
                        There has been ${renv} Build triggered from Jenkins, please confirm to proceed the build process.
                        <br/>
                        For confirmation, please login to Jenkins -> Go to the ${renv} Release Job for Data Science.
                        <br/>
                         ${env.RUN_DISPLAY_URL}
                        <br/>
                         -> Select the Checkbox and Click Proceed.
                    </p> 
                    <p>
                        Please note, the build will be timedout in 1 hr from invocation of Job.<br/><br/>
                
                        Job: ${env.JOB_NAME} <br/>
                        Build: ${env.BUILD_NUMBER} <br/>
                        More info at: ${env.RUN_DISPLAY_URL} 
                    </p>   
                </div>
            </body>
        </html>
    """
    
    msg
}

/**
 Send Slack alert to approvers 
*/
def sendBuildApproverAlert(approver) {
    def msg = """
                {
                    "attachments": [
                        {
                            "title": "Approval needed!",
                            "title_link": "${env.RUN_DISPLAY_URL}",
                            "text": "Wating for Approval, Job: ${env.JOB_NAME} # ${BUILD_NUMBER}",
                            "color": "#FFC107",
                            "fields": [
                                {
                                    "title": "Status",
                                    "value": "Wating for approval"
                                },
                                {
                                    "title": "Approver",
                                    "value": "${approver}"
                                }
                            ]
                        }
                    ]
                }
            """
    sendSlack(msg)
}

/**
* Publish the Egg
*/

def eggPublish(){
    if(deployment_env !='skip' && deployment_env !='dev') {
        withCredentials([file(credentialsId: "dx-ds-rnd-pypirc", variable: "pypirc_path")]) {
            sh(script: """
                #!/bin/bash
                sed -ie "s/${APP_VERSION}/${BUILD_VERSION}/g" version.py
                cp \${pypirc_path} ~/.pypirc
                make PUBLISH_EGG
            """)
        }
    }
}

/**
* Set Badge Status
*/

def setBadgeStatus(badgeConfig, status){
    badgeConfig.setStatus(status);
}

/**
* Update GH Pages
*/

def updateGHPages(){
    if(deployment_env =='prod')
    {
        //rename git reference so that it doesn't conflict with gh pages
        sh "mv .git .git_bkp"
        dir("temp-gh-pages"){
            checkout([$class: 'GitSCM', branches: [[name: "gh-pages"]],
                userRemoteConfigs: [[url: 'git@github.dx.utc.com:data-science-private/providence.git']]])
            sh "make -f ../Makefile  UPDATE_GH_PAGES"

        }
        //rename git backup reference so that Jenkins can submit the build historuy to git
        sh "mv .git_bkp .git"
    }
}