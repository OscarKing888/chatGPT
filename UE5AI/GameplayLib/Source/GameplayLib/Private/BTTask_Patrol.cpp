// BTTask_Patrol.cpp

#include "BTTask_Patrol.h"
#include "AIController.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "NavigationSystem.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Object.h"
#include "EnvironmentQuery/EnvQueryInstanceBlueprintWrapper.h"
#include "MonsterCharacter.h"



UBTTask_Patrol::UBTTask_Patrol()
{
    NodeName = TEXT("Patrol");
}

EBTNodeResult::Type UBTTask_Patrol::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    AAIController* AIController = OwnerComp.GetAIOwner();
    if (AIController == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    UBlackboardComponent* BlackboardComponent = AIController->GetBlackboardComponent();
    if (BlackboardComponent == nullptr)
    {
        return EBTNodeResult::Failed;
    }

	AMonsterCharacter* Monster = Cast<AMonsterCharacter>(AIController->GetPawn());
    if(!Monster)
	{
		return EBTNodeResult::Failed;
	}
    
    TArray<AActor*> PatrolPoints = Monster->PatrolPoints;

    if (PatrolPoints.Num() == 0)
    {
        return EBTNodeResult::Failed;
    }
    
    int32 CurrentPatrolIndex = BlackboardComponent->GetValueAsInt(GetSelectedBlackboardKey());
    if (PatrolPoints.Num() > 0 && PatrolPoints.IsValidIndex(CurrentPatrolIndex))
    {
        AActor* PatrolPoint = PatrolPoints[CurrentPatrolIndex];
        AIController->MoveToActor(PatrolPoint);

        // Update the index for the next patrol point
        int32 NextPatrolIndex = (CurrentPatrolIndex + 1) % PatrolPoints.Num();
        BlackboardComponent->SetValueAsInt(GetSelectedBlackboardKey(), NextPatrolIndex);

        return EBTNodeResult::Succeeded;
    }

    return EBTNodeResult::Failed;
}

