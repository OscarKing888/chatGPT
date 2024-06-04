// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BTTask_Roam.generated.h"

/**
 * 
 */
UCLASS()
class GAMEPLAYLIB_API UBTTask_Roam : public UBTTaskNode
{
	GENERATED_BODY()
	
public:	
	UBTTask_Roam();

protected:
	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
};



/**
 * 
 */
UCLASS()
class GAMEPLAYLIB_API UBTTask_MoveToTargetCustom : public UBTTaskNode
{
	GENERATED_BODY()
	
public:	
	UBTTask_MoveToTargetCustom();

protected:
	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
};
