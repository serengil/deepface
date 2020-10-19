#! groovy

pipeline {

  agent { label 'MeG' }

  environment {
    DOCKER_PULL_REPO='repo.eresearch.unimelb.edu.au:8000'
    DOCKER_PUSH_REPO='repo.eresearch.unimelb.edu.au:8001'
    DOCKER_IMAGE_NAME="happypet/webapp"
    GPU_DOCKER_IMAGE_NAME="happypet/webapp-gpu"
    BASE_IMAGE_NAME="happypet/tensorflow"
    BASE_IMAGE_TAG="1.15.2-py3"
    GPU_BASE_IMAGE_TAG="1.15.2-gpu-py3"
  }

  stages {
    stage('Pull base image'){
      steps {
        ansiColor('xterm') {
          script {
            docker.withRegistry("https://${env.DOCKER_PULL_REPO}",'repo-credentials') {
              script {
                docker.image("${env.BASE_IMAGE_NAME}:${env.BASE_IMAGE_TAG}").pull()
                docker.image("${env.BASE_IMAGE_NAME}:${env.GPU_BASE_IMAGE_TAG}").pull()
              }
            }
          }
        }
      }
    }

    stage('Build only'){
      when {
        not {
          anyOf {
            branch 'master'
            branch 'develop'
            branch 'release'
          }
        }
      }
      steps {
        ansiColor('xterm') {
          script {
            def version = sh(returnStdout: true, script:'git describe --tags --always').trim()
            docker.withRegistry("https://${env.DOCKER_PUSH_REPO}",'repo-credentials') {
              script {
                docker.build("${env.DOCKER_IMAGE_NAME}:${version}",'-f docker/release/Dockerfile .')
                docker.build("${env.GPU_DOCKER_IMAGE_NAME}:${version}",'-f docker/release_gpu/Dockerfile .')
              }
            }
          }
        }
      }
    }

    stage('Build and Push'){
      when {
        anyOf {
          branch 'master'
          branch 'develop'
          branch 'release'
        }
      }
      steps {
        ansiColor('xterm') {
          script {
            def version = sh(returnStdout: true, script:'git describe --tags --always').trim()
            docker.withRegistry("https://${env.DOCKER_PUSH_REPO}",'repo-credentials') {
              script {
                docker.build("${env.DOCKER_IMAGE_NAME}:${version}",'-f docker/release/Dockerfile .')
                  .push()
                docker.build("${env.DOCKER_IMAGE_NAME}:latest",'-f docker/release/Dockerfile .')
                  .push()
                docker.build("${env.GPU_DOCKER_IMAGE_NAME}:${version}",'-f docker/release_gpu/Dockerfile .')
                  .push()
                docker.build("${env.GPU_DOCKER_IMAGE_NAME}:latest",'-f docker/release_gpu/Dockerfile .')
                  .push()
              }
            }
          }
        }
      }
    }

    stage('Deploy'){
      when {
        anyOf {
          branch 'master'
          branch 'develop'
        }
      }
      steps {
        ansiColor('xterm') {
          script {
            sshagent (credentials: ['960a6936-d2d3-4d24-b9bb-c19e33f467ed']) {
              sh "ssh -o StrictHostKeyChecking=no happypet-real-dev.eresearch.unimelb.edu.au"
            }
          }
        }
      }
    }
  }
}
