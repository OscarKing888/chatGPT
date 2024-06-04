// BTTask_Roam.cpp

#include "BTTask_Roam.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "MonsterCharacter.h"

UBTTask_Roam::UBTTask_Roam()
{
    NodeName = TEXT("Roam");
}

EBTNodeResult::Type UBTTask_Roam::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    AAIController* AIController = OwnerComp.GetAIOwner();
    if (AIController == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    APawn* ControlledPawn = AIController->GetPawn();
    if (ControlledPawn == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    FVector Origin = ControlledPawn->GetActorLocation();
    UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(ControlledPawn->GetWorld());

    if (NavSys == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    FNavLocation RandomLocation;
    if (NavSys->GetRandomPointInNavigableRadius(Origin, 1000.0f, RandomLocation))
    {
        AIController->MoveToLocation(RandomLocation.Location);
        return EBTNodeResult::Succeeded;
    }

    return EBTNodeResult::Failed;
}


UBTTask_MoveToTargetCustom::UBTTask_MoveToTargetCustom()
{
    NodeName = TEXT("MoveToTargetCustom");
}

EBTNodeResult::Type UBTTask_MoveToTargetCustom::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    AAIController* AIController = OwnerComp.GetAIOwner();
    if (AIController == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    APawn* ControlledPawn = AIController->GetPawn();
    if (ControlledPawn == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(ControlledPawn->GetWorld());

    if (NavSys == nullptr)
    {
        return EBTNodeResult::Failed;
    }


    AMonsterCharacter* Monster = Cast<AMonsterCharacter>(ControlledPawn);
    if (!Monster || !Monster->MoveToTargetActor)
    {
        return EBTNodeResult::Failed;
    }

    // 如果有自定义的Navigation Query Filter，可以在这里应用
//     UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(GetWorld());
//     if (NavSys)
//     {
//         UNavigationQueryFilter* MyQueryFilter = NewObject<UNavigationQueryFilter>();
//         设置自定义过滤器的参数
//             MyQueryFilter->SetFilterParam(FNavAgentProperties::DefaultAgent, EQueryFilter::Cost::DefaultCost);
//         SetMoveQueryFilter(MyQueryFilter);        
//     }

    
    FVector Origin = Monster->MoveToTargetActor->GetActorLocation();
    
    FNavLocation RandomLocation;
    if(NavSys->ProjectPointToNavigation(Origin, RandomLocation))
    {
        AIController->MoveToLocation(RandomLocation.Location);
        return EBTNodeResult::Succeeded;
    }  

    return EBTNodeResult::Failed;
}

