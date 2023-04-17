// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/Tasks/BTTask_BlackboardBase.h"
#include "BTTask_Patrol.generated.h"

/**
 * 
 */
UCLASS()
class GAMEPLAYLIB_API UBTTask_Patrol : public UBTTask_BlackboardBase
{
	GENERATED_BODY()

public:

	UBTTask_Patrol();
	
protected:
	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;	
};
