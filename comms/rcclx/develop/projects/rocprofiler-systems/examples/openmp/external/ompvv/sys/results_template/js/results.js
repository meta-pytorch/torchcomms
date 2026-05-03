(function (angular) {
  
  'use strict';
  
  var ompvv = angular.module('ompvvSollve', ['ngRoute','ngAnimate', 'ngSanitize', 'ui.bootstrap', 'ansiToHtml']);

  ompvv.controller('ompvvSollveController', 
  ['$scope', '$http', '$log','$uibModal', '$timeout', 'ansi2html', '$filter', '$q',
    function($scope, $http, $log, $uibModal, $timeout, ansi2html, $filter, $q) {

      // Info message regarding table
      $scope.displayMessage = true;
      $timeout(function(){
        $scope.displayMessage = false;
      }, 5000)

      // var to show and hide results loading
      $scope.loadingResults = true;

      // Var to show and hide error message if no results were found
      $scope.errorMessage = false;

      // Table variables and getting the json file from the server
      $scope.tableContent = [];

      // Connecting to source code in the git repository
      $scope.SOLLVEgithubLink = "https://github.com/SOLLVE/sollve_vv"
      $scope.githubAPI = {};
      $scope.githubAPI.link = "https://api.github.com/";
      $scope.githubAPI.commits = "repos/SOLLVE/sollve_vv/commits/";

      $scope.gitCommits = {};

      // Function to load results
      $scope.loadResults = function(){
        if (typeof jsonResults == 'undefined'){
          $scope.loadingResults = false;
          $scope.errorMessage = true;
          $log.log("Error loading results json file");
          $scope.error = "Error loading results json file";
          return;
        }
        $scope.tableContent = jsonResults;
        angular.forEach($scope.tableContent, function(value, index) {
          var compilerName = value["Compiler name"];
          var testSystem = value["Test system"];
          var testName = value["Test name"];
          var testGitCommit = value["Test gitCommit"];
          var compilerResult = value["Compiler result"];
          var runtimeResult = value ["Runtime result"];
          // Adding test to list of all tests
          if ($scope.filters.testsOptions.indexOf(testName) == -1) {
            $scope.filters.testsOptions.push(testName);
          }
          // Adding compiler name to list of all compilers
          if ($scope.filters.compilerOptions.indexOf(compilerName) == -1) {
            $scope.filters.compilerOptions.push(compilerName);
          }
          // Adding system name to list of all systems
          if ($scope.filters.systemOptions.indexOf(testSystem) == -1) {
            $scope.filters.systemOptions.push(testSystem);
          }
          // Adding github commit
          if (testGitCommit != "" && !$scope.gitCommits.hasOwnProperty(testGitCommit)) {
            $scope.gitCommits[testGitCommit] = {};
            $scope.gitCommits[testGitCommit].isAccessible = false;
          }

          // Placing all results indexed by testname
          if ($scope.testsResults[testName] == undefined) {
            $scope.testsResults[testName] = {};
          }
          if ($scope.testsResults[testName][testSystem] == undefined) {
            $scope.testsResults[testName][testSystem] = {};
          }
          if ($scope.testFailed(compilerResult)) {
            $scope.testsResults[testName][testSystem][compilerName] = ["FAIL CE", index];
          } else if ($scope.testFailed(runtimeResult)) {
            $scope.testsResults[testName][testSystem][compilerName] = ["FAIL RE", index];
          } else {
            $scope.testsResults[testName][testSystem][compilerName] = ["PASS", index];
          }

          // Placing all compilers by test
          if ($scope.compilerByTest.get(testName) == undefined) {
            $scope.compilerByTest.set(testName, []);
          }
          if ($scope.compilerByTest.get(testName).indexOf(compilerName) == -1) {
            $scope.compilerByTest.get(testName).push(compilerName);
          }          
          // Placing all compilers by system
          if ($scope.compilerBySystems.get(testSystem) == undefined) {
            $scope.compilerBySystems.set(testSystem, []);
          }
          if ($scope.compilerBySystems.get(testSystem).indexOf(compilerName) == -1) {
            $scope.compilerBySystems.get(testSystem).push(compilerName);
          }

        });

        var allPromisses = [];
        angular.forEach($scope.gitCommits, function(value, key) {
          // See if we can find the corresponding commits in github
          allPromisses.push($http.get($scope.githubAPI["link"]+$scope.githubAPI["commits"]+key).then(
            function(data) {
              // Success
              $log.log("Commit " + key + " was found");
              value.isAccessible = true;
              return true;
            }, function(err) {
              // Failure
              $log.log("Commit " + key + " was not found");
              value.isAccessible = false;
              return false;
            })
          );// End of append
        }); // End of for each gitCommits

        // waiting for all commit checks
        $q.all(allPromisses).then(function(){
          $scope.loadingResults = false;
        });
      };

      $scope.testsResults = {};
      $scope.compilerBySystems = new Map();
      $scope.compilerByTest = new Map();

      // For the stats display
      $scope.getListOfSystems = function() {
        var returnList = [];
        $scope.compilerBySystems.forEach( function(value, key) {
            returnList.push(key);
        });
        return $filter('filterSystems')(returnList, $scope.filters);
      };
      $scope.getListOfCompilersSystems = function() {
        var returnList = [];
        $scope.compilerBySystems.forEach( function(compiler, system) {
          if ($filter('filterSystems')([system], $scope.filters).length != 0) {
            returnList.push.apply(returnList, compiler); 
          }
        });
        return $filter('filterCompilers')(returnList, $scope.filters);
      };

      $scope.getResultsSystemCompiler = function (test) {
        var returnList = [];
        $scope.compilerBySystems.forEach( function(systemCompilers, systemName) {
          if ($filter('filterSystems')([systemName], $scope.filters).length != 0) {
            var filteredSystemCompilers = $filter('filterCompilers')(systemCompilers, $scope.filters);
            filteredSystemCompilers.forEach(function (compiler) {
              var result=$scope.testsResults[test];
              result = result != undefined ? result[systemName]:undefined;
              result = result != undefined ? result[compiler]:undefined;
            
              returnList.push(result);
            });
          }
        });
        return returnList;
      }

      $scope.getListOfTests = function() {
        return $filter('filterTests')($scope.filters.testsOptions, $scope.filters);
      }

      // To sort results by column
      $scope.order = {};
      $scope.order.reverseSort = false;
      $scope.order.orderByField = 'Test path';
      $scope.colsToDisplay = ['Test name', 'Test system', 'Compiler name', 'Compiler result', 'Runtime result'];

      // To filter results by column
      $scope.filters = {};
      $scope.filters.testsOptions = [];
      $scope.filters.searchFilter = "";
      $scope.filters.compilerOptions = [];
      $scope.filters.compilerFilter = [];
      $scope.filters.systemOptions = [];
      $scope.filters.systemFilter = [];
      $scope.filters.compilerResultFilter = "Both";
      $scope.filters.runResultFilter = "Both";
      

      // Function for ordering table See here: 
      // https://stackoverflow.com/questions/23134773/angularjs-ngrepeat-orderby-when-property-key-has-spaces
      $scope.predicate = function(val) {
        return val[$scope.order.orderByField];
      }

      // function to verify if PASS is preset
      $scope.testPassed = function(result) {
        if (typeof result === 'string') {
          return result.indexOf("PASS") !== -1;
        } 
        return false;      }
      $scope.testFailed = function(result) {
        if (typeof result === 'string') {
          return result.indexOf("FAIL") !== -1;
        } 
        return false;
      }


      // More information results modal
      $scope.openResultModal = function (value) {
        var modalInstance = $uibModal.open({
          animation: true,
          component: 'modalResults',
          size: 'lg',
          resolve: {
            resultEntry: function() {
              return value;
            },
            ansi2html: function() {
              return ansi2html;
            },
            githubUrl: function() {
              return $scope.SOLLVEgithubLink;
            }, 
            showLink: function() {
              $log.log()
              if (value['Test gitCommit'] == '') return false;
              return $scope.gitCommits[value['Test gitCommit']].isAccessible;
            }

          }
        });

        modalInstance.result.then(function () {
          $log.info('Modal closed');
        }, function () {
          $log.info('Modal dismissed');
        });
      }
      $scope.loadResults();

  }]);

  var modalResultsCtrl = 

  ompvv.component('modalResults', {
    template: `
    <div class="modal-content">
    <div class="modal-header">
        <h3>
            <span class="label label-info" id="qid">{{$ctrl.resultEntry['Test name']}}</span> 
              <span ng-if="$ctrl.resultEntry['Test gitCommit']!=''" style="font-size:0.5em">
                <a ng-if="$ctrl.showLink" target="_blank" href="{{$ctrl.githubUrl+'/blob/'+$ctrl.resultEntry['Test gitCommit']+'/'+$ctrl.resultEntry['Test path']}}">
                  <span class="glyphicon glyphicon-file"> </span> 
                  source code  
                </a>
            </span>
        </h3>
        <h4>
            <span class="label label-primary">Path:</span><span>  {{$ctrl.resultEntry['Test path']}}</span> <br/>
        </h4>
        <h4 ng-if="$ctrl.resultEntry['Test gitCommit']!=''">
            <span class="label label-primary">Git Commit:</span><span>  {{$ctrl.resultEntry['Test gitCommit']}}</span> <br/>
        </h4>
        <h4>
            <span class="label label-primary">Compiler:</span><span>  {{$ctrl.resultEntry['Compiler name']}}</span>
        </h4>
    </div>
    <div class="modal-body">
        <div class="outputsContainer">
            <div style="margin-bottom: 15px" >
                <label class="btn btn-lg btn-block" 
                ng-click="$ctrl.compilerOutputActive=!$ctrl.compilerOutputActive"
                ng-class="{'btn-success': $ctrl.testPassed($ctrl.resultEntry['Compiler result']), 'btn-danger': $ctrl.testFailed($ctrl.resultEntry['Compiler result'])}">
                <i class="glyphicon glyphicon-chevron-right"></i>Compiler result = {{$ctrl.resultEntry['Compiler result']}}
                </label>
                <div class="slideDown" ng-show="$ctrl.compilerOutputActive">
                    <div class="well well-sm">
                            <strong>Compiler command: </strong>{{$ctrl.resultEntry['Compiler command']}}
                    </div>
                    <div class="well well-sm">
                            <strong>Compilation time range: </strong>{{$ctrl.resultEntry['Compiler starting date']}} - {{$ctrl.resultEntry['Compiler ending date']}}
                    </div>
                    <div class="well well-sm resultOutput">
                            <strong>Compilation output: </strong><br/><span class="outputText" ng-bind-html="$ctrl.color2html($ctrl.resultEntry['Compiler output'])"></span>
                    </div>
                </div>
            </div>
            <div style="margin-bottom: 15px" >
                <label class="btn btn-lg btn-primary btn-block" 
                ng-click="$ctrl.runtimeOutputActive=!$ctrl.runtimeOutputActive"
                ng-class="{'btn-success': $ctrl.testPassed($ctrl.resultEntry['Runtime result']), 'btn-danger': $ctrl.testFailed($ctrl.resultEntry['Runtime result'])}">
                        <i class="glyphicon glyphicon-chevron-right"></i>Runtime result = {{$ctrl.resultEntry['Runtime result']}}
                </label>
                <div class="slideDown" ng-show="$ctrl.runtimeOutputActive">
                    <div class="well well-sm">
                            <strong>Run binary path: </strong>{{$ctrl.resultEntry['Binary path']}}
                    </div>
                    <div class="well well-sm resultOutput">
                            <strong>Run output: </strong><br/><span class="outputText" ng-bind-html="$ctrl.color2html($ctrl.resultEntry['Runtime output'])"></span>
                    </div>
                </div>
            </div>            
        </div> 
    </div>
    <div class="modal-footer text-muted">
        <span id="answer"></span>
    `,
    controllerAs: '$ctrl',
    bindings: {
      resolve: '<',
      close: '&'
    },
    controller: function($sce) {
      var $ctrl = this;
      $ctrl.compilerOutputActive = false;
      $ctrl.runtimeOutputActive = false;
      $ctrl.$onInit = function () {
        $ctrl.resultEntry = $ctrl.resolve.resultEntry;
        $ctrl.ansi2html = $ctrl.resolve.ansi2html;
        $ctrl.githubUrl = $ctrl.resolve.githubUrl;
        $ctrl.showLink = $ctrl.resolve.showLink;
      }
      $ctrl.ok = function () {
        $uibModalInstance.close();
      };
    
      $ctrl.cancel = function () {
        $uibModalInstance.dismiss('cancel');
      };
  
      // function to verify if PASS is preset
      $ctrl.testPassed = function(result) {
        if (typeof result === 'string') {
          return result.indexOf("PASS") !== -1;
        } 
        return false;      }
      $ctrl.testFailed = function(result) {
        if (typeof result === 'string') {
          return result.indexOf("FAIL") !== -1;
        } 
        return false;
      }
      $ctrl.color2html = function(outputText) {
        return $sce.trustAsHtml($ctrl.ansi2html.toHtml(outputText).replace(/\n/g, '<br/>'));
      }
  
    }
  });

  ompvv.filter('filterCompilers', function(){
    return function(items, filters) {
      var filtered = [];
      angular.forEach(items, function(item){
        var containsCompiler = false;
        angular.forEach(filters.compilerFilter, function(compiler) {
          if (compiler == "" || item == compiler)
            containsCompiler = true;
        });
        if (containsCompiler || filters.compilerFilter == "") {
          filtered.push(item);
        }
      });
      return filtered;
    }
  })
  ompvv.filter('filterTests', function(){
    return function(items, filters) {
      var filtered = [];
      angular.forEach(items, function(item){
        if (filters.searchFilter == "" || item.includes(filters.searchFilter)) {
          filtered.push(item);
        }
      });
      return filtered;
    }
  })
  ompvv.filter('filterSystems', function(){
    return function(items, filters) {
      var filtered = [];
      angular.forEach(items, function(item){
        var containsSystem = false;
        angular.forEach(filters.systemFilter, function(system) {
          if (system == "" || item == system)
          containsSystem = true;
        });
        if (containsSystem || filters.systemFilter == "") {
          filtered.push(item);
        }
      });
      return filtered;
    }
  })
  
  ompvv.filter('applyResultsFilter', function() {
      return function (items, filters) {
        var filtered = [];
        if (filters.searchFilter == "" && filters.compilerFilter == "" && filters.systemFilter == "" && filters.compilerResultFilter == "Both" && filters.runResultFilter == "Both") {
          return items;
        }
        angular.forEach(items, function(item) {
          var removeItem = false;
          // filter by test name 
          if (filters.searchFilter != "" && !item['Test path'].includes(filters.searchFilter))
            removeItem = true;
          // Filter by compiler name 
          var containsCompiler = false;
          angular.forEach(filters.compilerFilter, function(compiler) {
            if (compiler != "" && item['Compiler name'] == compiler)
              containsCompiler = true;
          });
          if (filters.compilerFilter != "" && !containsCompiler)
            removeItem = true;
          // Filter by system name
          var containsSystem = false;
          angular.forEach(filters.systemFilter, function(system) {
            if (system != "" && item['Test system'] == system)
              containsSystem = true;
          });
          if (filters.systemFilter != "" && !containsSystem)
            removeItem = true;
          // Filter by compiler result
          if (filters.compilerResultFilter != "Both" && !item['Compiler result'].includes(filters.compilerResultFilter))
            removeItem = true;
          // Filter by Runtime result
          if (filters.runResultFilter != "Both" && !item['Runtime result'].includes(filters.runResultFilter))
            removeItem = true;
          if (!removeItem) 
            filtered.push(item);
        });
        return filtered;
      };
  });
  
})(window.angular);
